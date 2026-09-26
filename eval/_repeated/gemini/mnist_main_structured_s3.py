import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, Dataset
import os
from PIL import Image # Required for SyntheticMNISTDataset to create PIL images


# Preserve the original model architecture exactly
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # The input feature size (9216) is correct for MNIST (28x28)
        # after two conv layers, pooling, and flattening:
        # 1x28x28 -> Conv1 (32x26x26) -> Relu -> Conv2 (64x24x24) -> Relu -> MaxPool2d (64x12x12)
        # Flattened: 64 * 12 * 12 = 9216
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Synthetic data fallback for build_dataloader
class SyntheticMNISTDataset(Dataset):
    def __init__(self, num_samples=1000, transform=None):
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random 28x28 grayscale image (0-255 values)
        # and a random label (0-9)
        img_array = (torch.rand(28, 28) * 255).byte().numpy()
        image = Image.fromarray(img_array, mode='L') # Grayscale PIL Image

        label = torch.randint(0, 10, (1,)).item() # Single integer label

        if self.transform:
            image = self.transform(image)

        return image, label


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    # The Net class does not take any constructor arguments in its __init__ method.
    # If it did, model_kwargs from config.get("model_kwargs", {}) would be passed.
    return Net()


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = None
    if split in ["train", "val"]:
        try:
            # Attempt to load the real MNIST training dataset
            full_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

            # Define split ratios for train/val from the full_dataset
            # Default: 90% for training, 10% for validation
            train_ratio = config.get("data_split_ratio", {}).get("train", 0.9)
            val_ratio = 1.0 - train_ratio # Ensure ratios sum to 1.0

            num_train = int(len(full_dataset) * train_ratio)
            num_val = len(full_dataset) - num_train

            train_subset, val_subset = random_split(full_dataset, [num_train, num_val])

            if split == "train":
                dataset = train_subset
            elif split == "val":
                dataset = val_subset

        except FileNotFoundError as e:
            if allow_synthetic_data:
                print(f"WARNING: Real MNIST dataset not found at '{data_path}'. Using synthetic data. Error: {e}")
                dataset = SyntheticMNISTDataset(transform=transform)
            else:
                raise FileNotFoundError(
                    f"Real MNIST dataset not found at '{data_path}' and 'allow_synthetic_data' is False. "
                    f"Set 'allow_synthetic_data': True in config to enable synthetic fallback. "
                    f"Original error: {e}"
                )
        except Exception as e:
            # Catch other potential errors during data loading
            if allow_synthetic_data:
                print(f"WARNING: Error loading real MNIST dataset from '{data_path}'. Using synthetic data. Error: {e}")
                dataset = SyntheticMNISTDataset(transform=transform)
            else:
                raise RuntimeError(
                    f"Error loading real MNIST dataset from '{data_path}' and 'allow_synthetic_data' is False. "
                    f"Original error: {e}"
                )
    else:
        # The prompt specifically asks for "train" or "val" using random_split from a single dataset.
        raise ValueError(f"Unsupported split: '{split}'. This client module supports 'train' and 'val' splits only.")

    # DataLoader arguments (e.g., num_workers, pin_memory, shuffle) can be configured
    # via config.get("local", {}).get("dataloader_kwargs", {})
    dataloader_kwargs = config.get("local", {}).get("dataloader_kwargs", {})
    shuffle = dataloader_kwargs.get("shuffle", (split == "train")) # Shuffle for train, not for val
    num_workers = dataloader_kwargs.get("num_workers", 0) # Default to 0 workers for simplicity
    pin_memory = dataloader_kwargs.get("pin_memory", torch.cuda.is_available()) # Pin memory if CUDA available

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    """
    model.train() # Set the model to training mode

    data, target = batch

    # Move tensors to the device of the model parameters
    device = next(model.parameters()).device
    data, target = data.to(device), target.to(device)

    # Perform a forward pass
    output = model(data)
    loss = F.nll_loss(output, target)

    # Return the loss tensor. It still has the computational graph attached.
    return loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
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


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    return Net(**model_kwargs)


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

    try:
        # Attempt to load the real MNIST dataset
        # We download the training set and then split it into train/val
        full_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    except Exception as e:
        if allow_synthetic_data:
            print(f"WARNING: Could not load real MNIST dataset ({e}). Using synthetic data for split '{split}'.")
            # Generate synthetic data for fallback
            # MNIST images are 1x28x28, 10 classes
            num_samples = 1024
            synthetic_images = torch.randn(num_samples, 1, 28, 28)
            synthetic_labels = torch.randint(0, 10, (num_samples,))
            full_dataset = TensorDataset(synthetic_images, synthetic_labels)
        else:
            raise FileNotFoundError(
                f"MNIST dataset not found at '{data_path}' and 'allow_synthetic_data' is False. "
                "Please ensure the dataset is available or set 'allow_synthetic_data' to True for testing purposes."
            )

    # Split the full dataset into training and validation subsets
    # Using 90% for training and 10% for validation by default
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    if split == "train":
        dataset = train_dataset
        shuffle = True
    elif split == "val":
        dataset = val_dataset
        shuffle = False # Typically no shuffle for validation
    else:
        raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.get("local", {}).get("num_workers", 0),
        pin_memory=config.get("local", {}).get("pin_memory", False),
    )
    return dataloader


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    """
    data, target = batch
    
    # Move tensors to the device of the model parameters
    device = next(model.parameters()).device
    data, target = data.to(device), target.to(device)

    # Forward pass
    output = model(data)
    loss = F.nll_loss(output, target)

    return loss
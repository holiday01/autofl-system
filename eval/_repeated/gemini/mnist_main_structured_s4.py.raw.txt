import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import os

# Original Net class
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
    # The Net class does not currently take any constructor arguments,
    # so model_kwargs from config is not directly used here for Net itself.
    return Net()

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    local_config = config.get("local", {})
    batch_size = local_config.get("batch_size", 16)
    num_workers = local_config.get("num_workers", 0)
    pin_memory = local_config.get("pin_memory", False)

    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    val_ratio = config.get("val_ratio", 0.2) # Proportion of the training data to be used for local validation

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = None
    shuffle = False # Default shuffle
    
    try:
        # Attempt to load real MNIST data
        # For local train/val splits, we use the main training dataset (train=True)
        # and then divide it using random_split.
        full_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        
        # Determine lengths for train and val splits from the full dataset
        num_samples = len(full_dataset)
        num_val = int(num_samples * val_ratio)
        num_train = num_samples - num_val
        
        # Use a consistent generator for reproducible splits
        generator = torch.Generator().manual_seed(config.get("seed", 42))
        train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val], generator=generator)

        if split == "train":
            dataset = train_dataset
            shuffle = True # Shuffle training data
        elif split == "val":
            dataset = val_dataset
            shuffle = False # Typically no shuffle for validation data to ensure consistent evaluation
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    except Exception as e:
        if allow_synthetic_data:
            print(f"Warning: Could not load real MNIST data from '{data_path}'. Error: {e}. Using synthetic data.")
            
            # Define synthetic data parameters
            synthetic_num_samples = 1000 # A reasonable total size for synthetic data
            synthetic_img_shape = (1, 28, 28)
            synthetic_num_classes = 10

            class SyntheticMNISTDataset(Dataset):
                def __init__(self, num_samples, img_shape, num_classes, transform=None, seed=42):
                    self.num_samples = num_samples
                    # Use a separate generator for synthetic data generation for consistency
                    gen_data = torch.Generator().manual_seed(seed + 1) 
                    self.data = torch.randn(num_samples, *img_shape, generator=gen_data)
                    self.targets = torch.randint(0, num_classes, (num_samples,), generator=gen_data)
                    self.transform = transform

                def __len__(self):
                    return self.num_samples

                def __getitem__(self, idx):
                    img, target = self.data[idx], self.targets[idx]
                    if self.transform:
                        # transforms.ToTensor() will be idempotent on a tensor.
                        # transforms.Normalize will work on a tensor.
                        img = self.transform(img)
                    return img, target
            
            # Create full synthetic dataset
            full_synthetic_dataset = SyntheticMNISTDataset(
                synthetic_num_samples, synthetic_img_shape, synthetic_num_classes, 
                transform=transform, seed=config.get("seed", 42)
            )

            # Split synthetic dataset
            synthetic_num_val = int(synthetic_num_samples * val_ratio)
            synthetic_num_train = synthetic_num_samples - synthetic_num_val
            
            # Use the same generator for splitting synthetic data for consistency
            generator_synthetic = torch.Generator().manual_seed(config.get("seed", 42))
            train_synthetic_dataset, val_synthetic_dataset = random_split(
                full_synthetic_dataset, 
                [synthetic_num_train, synthetic_num_val],
                generator=generator_synthetic
            )
            
            if split == "train":
                dataset = train_synthetic_dataset
                shuffle = True
            elif split == "val":
                dataset = val_synthetic_dataset
                shuffle = False
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

        else:
            # If real data is unavailable and synthetic data is not allowed, raise an error
            raise FileNotFoundError(
                f"Failed to load MNIST dataset from '{data_path}'. "
                f"If you want to use synthetic data instead, set 'allow_synthetic_data: True' in your config. "
                f"Original error: {e}"
            )

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=True # Useful for FL to ensure consistent batch sizes
    )

def train_step(model: torch.nn.Module, batch, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    """
    model.train() # Ensure model is in training mode
    
    # Move batch to model's device
    device = next(model.parameters()).device
    data, target = batch
    data, target = data.to(device), target.to(device)

    # Perform forward pass
    output = model(data)
    
    # Calculate loss
    loss = F.nll_loss(output, target)
    
    return loss
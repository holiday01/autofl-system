import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import os

# Preserve original model architecture (Keras to PyTorch conversion)
class MNISTConvNet(nn.Module):
    """
    A PyTorch implementation of the original Keras Simple MNIST convnet.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        # Calculate input features for the dense layer based on original Keras architecture
        # Input: (batch_size, 1, 28, 28)
        # Conv1 output shape: (batch_size, 32, 26, 26)  (28 - 3 + 1 = 26)
        # Pool1 output shape: (batch_size, 32, 13, 13)
        # Conv2 output shape: (batch_size, 64, 11, 11)  (13 - 3 + 1 = 11)
        # Pool2 output shape: (batch_size, 64, 5, 5)   (floor(11/2) = 5)
        # Flattened shape: 64 * 5 * 5 = 1600
        self.dense = nn.Linear(1600, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        # Note: No softmax here. nn.CrossEntropyLoss expects raw logits.
        return x

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the PyTorch MNISTConvNet model.

    Args:
        config (dict): Configuration dictionary, potentially containing "model_kwargs".

    Returns:
        torch.nn.Module: The instantiated MNISTConvNet model.
    """
    model_kwargs = config.get("model_kwargs", {})
    num_classes = model_kwargs.get("num_classes", 10)
    return MNISTConvNet(num_classes=num_classes)

# --- Synthetic Data Fallback ---
class SyntheticMNISTDataset(Dataset):
    """
    A synthetic MNIST-like dataset for testing when real data is unavailable.
    Generates random images and labels.
    """
    def __init__(self, num_samples: int = 1000, img_shape=(1, 28, 28), num_classes: int = 10):
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image data (float32, [0, 1])
        image = torch.randn(self.img_shape)
        # Generate random class label (long integer for CrossEntropyLoss)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

# Global variable to store train/val split datasets for consistency across calls
_DATASET_SPLITS = {}

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    Splits the MNIST training dataset into train/validation subsets using random_split.

    Args:
        config (dict): Configuration dictionary, including "local", "data_path",
                       "allow_synthetic_data", and "validation_split_ratio".
        split (str): The desired data split ("train" or "val").

    Returns:
        DataLoader: A PyTorch DataLoader for the specified split.

    Raises:
        FileNotFoundError: If real data is unavailable and synthetic data is not allowed.
        ValueError: If an invalid split name is requested.
    """
    global _DATASET_SPLITS

    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    # Default validation_split_ratio to 0.1, as used in the original Keras script
    validation_split_ratio = config.get("validation_split_ratio", 0.1)

    # Define transforms for MNIST images
    transform = transforms.Compose([
        transforms.ToTensor(), # Scales images to [0, 1] and converts to CHW format
    ])

    # Perform the train/val split only once per client process
    if "full_dataset_split_done" not in _DATASET_SPLITS:
        try:
            # Attempt to load the real MNIST training dataset
            # If `train=True`, `MNIST` will use the training set.
            # `download=True` will download if not available.
            full_dataset = MNIST(root=data_path, train=True, download=True, transform=transform)
            print(f"Successfully loaded real MNIST dataset from {data_path}.")

        except Exception as e:
            if allow_synthetic_data:
                print(f"Warning: Could not load real MNIST dataset ({e}). Using synthetic data.")
                # Create a synthetic dataset if real data fails and synthetic is allowed
                full_dataset = SyntheticMNISTDataset(num_samples=60000, img_shape=(1, 28, 28), num_classes=10)
                print(f"Successfully created synthetic MNIST dataset with {len(full_dataset)} samples.")
            else:
                raise FileNotFoundError(
                    f"Failed to load MNIST dataset from '{data_path}' and `allow_synthetic_data` is False. "
                    "Please ensure the data path is correct or set `allow_synthetic_data` to True for fallback."
                ) from e

        # Use a fixed generator seed for reproducible random_split across calls/clients
        # This ensures the 'train' and 'val' splits are always the same.
        generator = torch.Generator().manual_seed(42)

        num_total_samples = len(full_dataset)
        num_val_samples = int(validation_split_ratio * num_total_samples)
        num_train_samples = num_total_samples - num_val_samples

        if num_train_samples <= 0 or num_val_samples <= 0:
            raise ValueError(f"Not enough samples to create train/val split with {validation_split_ratio} ratio. "
                             f"Total samples: {num_total_samples}, Train samples: {num_train_samples}, "
                             f"Val samples: {num_val_samples}")

        # Perform the random split
        train_subset, val_subset = random_split(
            full_dataset, [num_train_samples, num_val_samples], generator=generator
        )
        _DATASET_SPLITS["train"] = train_subset
        _DATASET_SPLITS["val"] = val_subset
        _DATASET_SPLITS["full_dataset_split_done"] = True # Mark that splitting is done
        print(f"Dataset split: Train samples = {len(train_subset)}, Validation samples = {len(val_subset)}")

    # Retrieve the appropriate subset based on the 'split' argument
    dataset = _DATASET_SPLITS.get(split)
    if dataset is None:
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'val'.")

    # Return a DataLoader with shuffling only for the training split
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))

def train_step(model: torch.nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Runs ONE forward pass and returns the loss tensor with gradient attached.
    Does NOT call loss.backward() or optimizer.step().

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        batch: A batch of data (inputs, labels) from the DataLoader.
        optimizer: The optimizer (not used directly here, but part of signature).
        config (dict): Configuration dictionary (not used directly here, but part of signature).

    Returns:
        torch.Tensor: The loss tensor with gradient attached.
    """
    inputs, labels = batch

    # Determine the device of the model parameters
    # Assuming all model parameters are on the same device
    device = next(model.parameters()).device

    # Move inputs and labels to the model's device
    inputs = inputs.to(device)
    # For nn.CrossEntropyLoss, labels should be class indices (long integers),
    # not one-hot encoded. The torchvision MNIST dataset provides integer labels.
    labels = labels.to(device)

    # Perform a single forward pass
    outputs = model(inputs)

    # Calculate the loss
    # Keras's `categorical_crossentropy` with a final `softmax` activation
    # is equivalent to PyTorch's `nn.CrossEntropyLoss` with raw logits.
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    return loss
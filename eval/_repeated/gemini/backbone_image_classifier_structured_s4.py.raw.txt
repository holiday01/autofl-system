from os import path
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset

# Conditional imports for torchvision components
# We keep lightning.pytorch.utilities.imports to check _TORCHVISION_AVAILABLE
# even though the rest of lightning.pytorch is not used.
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
    from torchvision.datasets import MNIST
    _TRANSFORM_TO_TENSOR = transforms.ToTensor()
else:
    # If torchvision is not available, we cannot use MNIST directly.
    # Define a placeholder transform for consistency, though it won't be used for real data.
    class _PlaceholderToTensor:
        def __call__(self, img):
            # For synthetic data, input is already a tensor, so just return.
            if isinstance(img, torch.Tensor):
                return img
            # Basic conversion for common image types if it were to be called on non-tensor inputs
            # For MNIST, this would typically involve PIL Image.
            # Example: from PIL import Image; return torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255.0
            # but for this FL setup, we expect real data loading to fail without torchvision.
            raise NotImplementedError("Cannot convert non-tensor image to tensor without torchvision.")

    _TRANSFORM_TO_TENSOR = _PlaceholderToTensor()


class Backbone(torch.nn.Module):
    """
    >>> Backbone()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Backbone(
      (l1): Linear(...)
      (l2): Linear(...)
    )
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


class SyntheticMNIST(Dataset):
    """A synthetic dataset that mimics the shape and type of MNIST data."""
    def __init__(self, num_samples=1000, img_shape=(1, 28, 28), num_classes=10):
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.num_classes = num_classes
        # Generate random images (float tensors)
        self.data = torch.randn(num_samples, *img_shape)
        # Generate random labels (long tensors)
        self.targets = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.

    Args:
        config: A dictionary containing configuration parameters.
                Model-specific arguments are expected under "model_kwargs".

    Returns:
        A torch.nn.Module instance.
    """
    model_kwargs = config.get("model_kwargs", {})
    return Backbone(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Args:
        config: A dictionary containing configuration parameters.
                Expected keys:
                - "local": dict with "batch_size" (int, default 16)
                - "data_path": str (default ".")
                - "allow_synthetic_data": bool (default False)
        split: The name of the data split to load ("train" or "val").

    Returns:
        A DataLoader for the specified split.

    Raises:
        ValueError: If an invalid split name is provided.
        FileNotFoundError: If real data cannot be loaded and synthetic data is not allowed.
        RuntimeError: If the dataset initialization logic fails unexpectedly.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    dataset = None

    if _TORCHVISION_AVAILABLE:
        try:
            # Load the full training dataset first, then split for train/val
            full_train_dataset = MNIST(root=data_path, train=True, download=True, transform=_TRANSFORM_TO_TENSOR)

            # Split the training dataset into train and validation parts, mirroring the original script
            train_size = 55000
            val_size = 5000
            if len(full_train_dataset) != (train_size + val_size):
                raise ValueError(
                    f"Expected MNIST train dataset of size {train_size + val_size}, "
                    f"but got {len(full_train_dataset)}. "
                    "Cannot perform specified train/val split. "
                    "Ensure MNIST data is correctly downloaded or specify correct sizes."
                )

            train_dataset, val_dataset = random_split(
                full_train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # Use fixed seed as in original
            )

            if split == "train":
                dataset = train_dataset
            elif split == "val":
                dataset = val_dataset
            else:
                raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

        except Exception as e:
            if allow_synthetic_data:
                print(f"WARNING: Could not load real MNIST dataset (Error: {e}). Using synthetic data for split '{split}'.")
                if split == "train":
                    dataset = SyntheticMNIST(num_samples=1000)
                elif split == "val":
                    dataset = SyntheticMNIST(num_samples=100)
                else:
                    raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")
            else:
                raise FileNotFoundError(
                    f"Failed to load real MNIST dataset from {data_path}. "
                    f"Original error: {e}. "
                    "Set 'allow_synthetic_data' to True in the config to use synthetic data instead."
                ) from e
    else:  # _TORCHVISION_AVAILABLE is False
        if allow_synthetic_data:
            print(f"WARNING: torchvision is not available. Using synthetic data for split '{split}'.")
            if split == "train":
                dataset = SyntheticMNIST(num_samples=1000)
            elif split == "val":
                dataset = SyntheticMNIST(num_samples=100)
            else:
                raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")
        else:
            raise FileNotFoundError(
                "torchvision is not available, and 'allow_synthetic_data' is False. "
                "Cannot load MNIST dataset without torchvision."
            )

    if dataset is None:
        raise RuntimeError(f"Dataset for split '{split}' could not be initialized. This indicates an internal logic error.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: torch.nn.Module, batch, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.

    Args:
        model: The PyTorch model.
        batch: A batch of data (inputs, targets).
        optimizer: The optimizer (not used for step/backward here, but typically passed).
        config: A dictionary containing configuration parameters.

    Returns:
        The loss tensor with gradients attached.
    """
    # Move batch to the same device as the model
    device = next(model.parameters()).device

    x, y = batch
    x, y = x.to(device), y.to(device)

    # Forward pass
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)

    return loss
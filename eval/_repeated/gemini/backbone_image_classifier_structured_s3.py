from os import path
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset

from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
    from lightning.pytorch.demos.mnist_datamodule import MNIST


class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


class SyntheticMNISTDataset(Dataset):
    def __init__(self, num_samples=60000, img_shape=(1, 28, 28), num_classes=10):
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, *img_shape)
        self.targets = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def build_model(config: dict) -> torch.nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    # The LitClassifier wraps the Backbone, which is the actual model architecture.
    # We directly instantiate Backbone as the FL model.
    return Backbone(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    if not _TORCHVISION_AVAILABLE:
        if allow_synthetic_data:
            print("WARNING: torchvision is not available. Using synthetic data for MNIST.")
            full_dataset = SyntheticMNISTDataset(num_samples=60000)  # Mimic MNIST train set size
        else:
            raise FileNotFoundError(
                "torchvision is not available, cannot load MNIST data. Set 'allow_synthetic_data: true' "
                "in config to use synthetic data."
            )
    else:
        transform = transforms.ToTensor()
        try:
            # Attempt to load the real MNIST dataset
            full_dataset = MNIST(root=data_path, train=True, download=True, transform=transform)
        except Exception as e:
            if allow_synthetic_data:
                print(f"WARNING: Could not load real MNIST data from {data_path} (error: {e}). Using synthetic data.")
                full_dataset = SyntheticMNISTDataset(num_samples=60000)  # Mimic MNIST train set size
            else:
                raise FileNotFoundError(
                    f"Failed to load MNIST dataset from {data_path}. Set 'allow_synthetic_data: true' "
                    "in config to use synthetic data, or ensure data is available/downloadable. "
                    f"Original error: {e}"
                )

    # Replicate the splitting logic from MyDataModule in the original script
    # Original MNIST train dataset size is 60000 samples.
    # Original split: 55000 train, 5000 val
    original_train_split_ratio = 55000 / 60000.0

    total_samples = len(full_dataset)
    train_size = int(total_samples * original_train_split_ratio)
    val_size = total_samples - train_size  # Ensure sum matches total_samples

    if total_samples < 60000:
        print(f"WARNING: Dataset size ({total_samples}) is smaller than standard MNIST training set (60000). "
              f"Adjusting train/val split proportionally to {train_size}/{val_size}.")

    # Use the same generator for reproducibility as in the original script
    generator = torch.Generator().manual_seed(42)
    mnist_train, mnist_val = random_split(full_dataset, [train_size, val_size], generator=generator)

    if split == "train":
        dataset = mnist_train
    elif split == "val":
        dataset = mnist_val
    else:
        raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'val'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    # Get the device from the model parameters
    device = next(model.parameters()).device

    # Move batch tensors to the model's device
    x, y = batch
    x, y = x.to(device), y.to(device)

    # Perform a single forward pass
    y_hat = model(x)

    # Calculate the loss using cross_entropy, as defined in LitClassifier's training_step
    loss = F.cross_entropy(y_hat, y)

    # Return the loss tensor with gradients attached
    return loss
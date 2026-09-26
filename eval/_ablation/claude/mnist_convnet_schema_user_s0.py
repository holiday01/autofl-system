import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms


class MNISTConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.5)
        # Input 28x28 → conv1(26x26) → pool(13x13) → conv2(11x11) → pool(5x5)
        self.fc = nn.Linear(64 * 5 * 5, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return MNISTConvNet(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    transform = transforms.Compose([transforms.ToTensor()])

    try:
        full_dataset = torchvision.datasets.MNIST(
            root=data_path, train=True, download=False, transform=transform
        )
        n_total = len(full_dataset)
        n_val = max(1, int(0.1 * n_total))
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
    except Exception:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset not found at '{data_path}'. "
                "Pre-download the dataset or set config['allow_synthetic_data'] = True "
                "to use synthetic data for testing purposes only."
            )
        n_samples = 200
        x = torch.randn(n_samples, 1, 28, 28)
        y = torch.randint(0, 10, (n_samples,))
        full_dataset = TensorDataset(x, y)
        n_val = max(1, int(0.1 * n_samples))
        n_train = n_samples - n_val
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    chosen = train_dataset if split == "train" else val_dataset
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    logits = model(images)
    return F.cross_entropy(logits, labels)
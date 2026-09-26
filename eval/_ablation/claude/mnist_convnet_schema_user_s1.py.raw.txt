import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

try:
    import keras
    from keras import layers
except ImportError:
    keras = None
    layers = None

num_classes = 10
input_shape = (28, 28, 1)


class MNISTConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64 * 5 * 5, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


def build_model(config: dict) -> nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    return MNISTConvNet(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = datasets.MNIST(
            root=data_path, train=True, download=False, transform=transform
        )
    except Exception:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset not found at '{data_path}'. "
                "Ensure the data exists at data_path or set "
                "config['allow_synthetic_data'] = True to use synthetic data."
            )
        x = torch.randn(1000, 1, 28, 28)
        y = torch.randint(0, 10, (1000,))
        full_dataset = TensorDataset(x, y)

    n_total = len(full_dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
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
    loss = F.cross_entropy(logits, labels)
    return loss
from os import path
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


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


def build_model(config):
    hidden_dim = config.get("hidden_dim", 128)
    return Backbone(hidden_dim=hidden_dim)


def build_dataloader(config, split="train"):
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    full_dataset = MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    train_set, val_set = random_split(
        full_dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
    )

    if split == "train":
        return DataLoader(train_set, batch_size=batch_size)
    return DataLoader(val_set, batch_size=batch_size)


def train_step(model, batch, optimizer, config):
    x, y = batch
    optimizer.zero_grad()
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    return loss
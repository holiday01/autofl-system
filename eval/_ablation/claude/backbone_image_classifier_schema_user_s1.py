from os import path
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

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


def build_model(config: dict) -> torch.nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    return Backbone(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    dataset = None
    load_error = None

    if _TORCHVISION_AVAILABLE:
        try:
            transform = transforms.ToTensor()
            full_dataset = MNIST(data_path, train=True, download=False, transform=transform)
            total = len(full_dataset)
            train_size = int(0.9 * total)
            val_size = total - train_size
            train_subset, val_subset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            dataset = train_subset if split == "train" else val_subset
        except Exception as exc:
            load_error = exc
    else:
        load_error = ImportError("torchvision is not available")

    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real MNIST dataset could not be loaded from '{data_path}' "
                f"(reason: {load_error}). "
                "Set config['allow_synthetic_data'] = True to fall back to synthetic data."
            )
        n = 1000 if split == "train" else 200
        x = torch.randn(n, 1, 28, 28)
        y = torch.randint(0, 10, (n,))
        dataset = TensorDataset(x, y)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: torch.nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    return loss
"""
FL client for backbone_image_classifier.py (Lightning MNIST backbone classifier).
Structured conversion: extracts Backbone from LightningModule into plain PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class Backbone(nn.Module):
    """Extracted from LitClassifier — plain PyTorch, no Lightning dependency."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return Backbone(hidden_dim=kwargs.get("hidden_dim", 128))


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        full_ds = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        n_val = max(1, int(0.1 * len(full_ds)))
        train_ds, val_ds = random_split(
            full_ds, [len(full_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_ds if split == "train" else val_ds
    except Exception:
        n = 1000
        data = torch.randn(n, 1, 28, 28)
        targets = torch.randint(0, 10, (n,))
        full_ds = TensorDataset(data, targets)
        n_val = max(1, int(0.1 * n))
        train_ds, val_ds = random_split(
            full_ds, [n - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_ds if split == "train" else val_ds

    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """One forward pass — returns CrossEntropyLoss with grad_fn attached.
    The FL runtime is responsible for loss.backward() and optimizer.step().
    """
    device = next(model.parameters()).device
    x, y = batch[0].to(device), batch[1].to(device)
    y_hat = model(x)
    return F.cross_entropy(y_hat, y)

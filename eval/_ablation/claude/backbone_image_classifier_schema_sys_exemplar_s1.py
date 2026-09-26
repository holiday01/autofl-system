"""
Auto-generated FL client module.
Original script: backbone_image_classifier.py (Lightning AI MNIST backbone example)

Exposes:
  build_model(config)                    -> nn.Module
  build_dataloader(config, split)        -> DataLoader
  train_step(model, batch, opt, config)  -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""

from os import path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from torchvision import transforms
    from torchvision.datasets import MNIST as TorchvisionMNIST
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


# ── Model (extracted from LitClassifier.backbone) ───────────────────────

class Backbone(nn.Module):
    """
    >>> Backbone()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Backbone(
      (l1): Linear(...)
      (l2): Linear(...)
    )
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


# ── FL Interface ─────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return Backbone(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local        = config.get("local", {})
    batch_size   = local.get("batch_size",  config.get("batch_size",  16))
    num_workers  = local.get("num_workers", config.get("num_workers", 2))
    pin_memory   = local.get("pin_memory",  True)

    data_path  = config.get("data_path", ".")
    val_ratio  = config.get("val_ratio",  0.1)
    seed       = config.get("seed",       42)

    full_dataset = None

    if _TORCHVISION_AVAILABLE:
        try:
            transform    = transforms.ToTensor()
            full_dataset = TorchvisionMNIST(
                data_path, train=True, download=True, transform=transform
            )
        except Exception:
            full_dataset = None

    if full_dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset could not be loaded from '{data_path}' "
                "(torchvision may be missing or the download failed). "
                "Set config['allow_synthetic_data'] = True to use synthetic "
                "data for testing, or supply a valid data_path containing the "
                "MNIST dataset."
            )
        # Synthetic fallback: (N, 1, 28, 28) float images, labels in [0, 9].
        # Backbone.forward handles arbitrary leading dims via x.view(B, -1).
        n_samples    = 500
        X            = torch.randn(n_samples, 1, 28, 28)
        y            = torch.randint(0, 10, (n_samples,))
        full_dataset = TensorDataset(X, y)

    n_val   = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    ds = train_ds if split == "train" else val_ds
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        x, y  = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        x = batch.get("input", batch.get("x", batch.get("image")))
        y = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    y_hat = model(x)
    loss  = F.cross_entropy(y_hat, y)
    return loss
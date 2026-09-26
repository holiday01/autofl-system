import argparse
from sklearn import svm, metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ── PyTorch equivalent of sklearn SVC(kernel="rbf") ──────────────────────────
# The sklearn SVC has no nn.Module representation, so we replace it with a
# two-hidden-layer MLP whose non-linear capacity approximates an RBF kernel SVM.
# Input: 4 Iris features    Output: 3 class logits

class SVMClassifier(nn.Module):
    """
    Feedforward classifier that replaces sklearn SVC with an RBF kernel.
    Two hidden layers with BatchNorm + ReLU provide the non-linear decision
    boundary that the RBF kernel would otherwise supply.
    """

    def __init__(
        self,
        in_features: int = 4,
        num_classes: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── FL API ────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the SVMClassifier.

    Recognised model_kwargs:
      in_features  (int, default 4)   – number of input features
      num_classes  (int, default 3)   – number of output classes
      hidden_dim   (int, default 64)  – width of each hidden layer
    """
    kwargs = config.get("model_kwargs", {})
    return SVMClassifier(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the Iris dataset (or a synthetic fallback).

    Config keys consumed:
      config["local"]["batch_size"]  – mini-batch size (default 16)
      config["data_path"]            – unused for Iris; kept for API parity
      config["seed"]                 – RNG seed for the train/val split (default 42)
    """
    batch_size: int = config.get("local", {}).get("batch_size", 16)
    _data_path: str = config.get("data_path", ".")   # noqa: F841 (Iris is built-in)
    seed: int = config.get("seed", 42)

    # ── Real dataset ──────────────────────────────────────────────────────────
    try:
        iris = load_iris()
        X = torch.tensor(iris.data, dtype=torch.float32)   # (150, 4)
        y = torch.tensor(iris.target, dtype=torch.long)    # (150,)

        # Replicate sklearn StandardScaler: zero-mean, unit-variance per feature
        mean = X.mean(dim=0)
        std  = X.std(dim=0).clamp(min=1e-8)
        X = (X - mean) / std

        dataset: TensorDataset = TensorDataset(X, y)

    # ── Synthetic fallback ────────────────────────────────────────────────────
    except Exception:
        X = torch.randn(150, 4)
        y = torch.randint(0, 3, (150,))
        dataset = TensorDataset(X, y)

    # ── Reproducible 80 / 20 train–val split ─────────────────────────────────
    n_total  = len(dataset)
    n_val    = max(1, int(round(0.2 * n_total)))
    n_train  = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    chosen_ds = train_ds if split == "train" else val_ds
    return DataLoader(
        chosen_ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=False,
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,           # provided by FL runtime; NOT called here
    config: dict,
) -> torch.Tensor:
    """
    Single forward pass.  Returns the loss tensor with gradients attached.
    backward() and optimizer.step() are intentionally omitted — the FL
    runtime is responsible for those.
    """
    device = next(model.parameters()).device

    X, y = batch
    X = X.to(device)
    y = y.to(device)

    logits = model(X)                          # (B, num_classes)
    loss   = nn.CrossEntropyLoss()(logits, y)  # scalar, grad_fn intact

    return loss  # ← DO NOT call loss.backward() or optimizer.step() here
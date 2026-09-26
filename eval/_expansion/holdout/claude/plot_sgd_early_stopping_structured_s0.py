import sys
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import shuffle
from sklearn.utils._testing import ignore_warnings


# ── Model ──────────────────────────────────────────────────────────────────────
# Equivalent to sklearn.linear_model.SGDClassifier: a single linear layer
# trained with BCEWithLogitsLoss for binary classification.

class SGDLinearClassifier(nn.Module):
    """Single-layer linear binary classifier, equivalent to SGDClassifier."""

    def __init__(self, in_features: int = 784):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logit (shape: [batch]), compatible with BCEWithLogitsLoss.
        return self.linear(x).squeeze(-1)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_mnist_tensors(
    data_path: str = ".",
    n_samples=None,
    class_0: str = "0",
    class_1: str = "8",
):
    """Load MNIST binary subset from OpenML; cache under *data_path*.

    Returns
    -------
    X : torch.FloatTensor  shape (N, 784)
    y : torch.FloatTensor  shape (N,)   — 0.0 for class_0, 1.0 for class_1
    """
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=data_path,
    )

    mask = np.logical_or(
        mnist.target.astype(str) == str(class_0),
        mnist.target.astype(str) == str(class_1),
    )

    X, y = shuffle(mnist.data[mask], mnist.target[mask], random_state=42)
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]

    y_bin = (y.astype(str) == str(class_1)).astype(np.float32)
    X_f32 = X.astype(np.float32)

    return torch.from_numpy(X_f32), torch.from_numpy(y_bin)


# ── FL API ─────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the SGDLinearClassifier.

    Recognised model_kwargs
    -----------------------
    in_features : int  — input dimensionality (default 784 for MNIST).
    """
    kwargs = config.get("model_kwargs", {})
    in_features = kwargs.get("in_features", 784)
    return SGDLinearClassifier(in_features=in_features)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Config keys consumed
    --------------------
    local.batch_size          : int   (default 16)
    data_path                 : str   (default '.')  — OpenML cache directory
    val_fraction              : float (default 0.2)
    allow_synthetic_data      : bool  (default False)
    model_kwargs.in_features  : int   (default 784)
    model_kwargs.class_0      : str   (default '0')
    model_kwargs.class_1      : str   (default '8')
    model_kwargs.n_samples    : int   (default None — use full filtered set)
    """
    local_cfg   = config.get("local", {})
    batch_size  = local_cfg.get("batch_size", 16)
    data_path   = config.get("data_path", ".")
    val_frac    = config.get("val_fraction", 0.2)

    mkw       = config.get("model_kwargs", {})
    class_0   = mkw.get("class_0",   "0")
    class_1   = mkw.get("class_1",   "8")
    n_samples = mkw.get("n_samples", None)
    in_feats  = mkw.get("in_features", 784)

    # ── Try to load real data ──────────────────────────────────────────────────
    try:
        X_t, y_t = _load_mnist_tensors(
            data_path=data_path,
            n_samples=n_samples,
            class_0=class_0,
            class_1=class_1,
        )
        dataset = TensorDataset(X_t, y_t)

    except Exception as exc:
        # ── Synthetic fallback — only if explicitly permitted ──────────────────
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real MNIST data could not be loaded from '{data_path}' and "
                f"config['allow_synthetic_data'] is False. "
                f"Set allow_synthetic_data=True to fall back to random tensors "
                f"(for smoke-testing only). Original error: {exc}"
            ) from exc

        n_syn = n_samples if n_samples else 1000
        X_syn = torch.randn(n_syn, in_feats)
        y_syn = torch.randint(0, 2, (n_syn,)).float()
        dataset = TensorDataset(X_syn, y_syn)

    # ── Train / val split ─────────────────────────────────────────────────────
    total      = len(dataset)
    val_size   = max(1, int(total * val_frac))
    train_size = total - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

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
    optimizer,          # accepted but NOT called — FL runtime owns backward/step
    config: dict,
) -> torch.Tensor:
    """One forward pass; returns the loss tensor with grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do so.
    """
    device = next(model.parameters()).device

    X, y = batch
    X = X.to(device)
    y = y.to(device)

    logits = model(X)                        # shape: (batch,)
    loss   = nn.BCEWithLogitsLoss()(logits, y)
    return loss                              # grad still attached; no .backward()
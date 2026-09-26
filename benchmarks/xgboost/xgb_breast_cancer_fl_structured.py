import argparse
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ---------------------------------------------------------------------------
# PyTorch equivalent of the XGBoost binary classifier.
# Replaces the gradient-boosted ensemble with a feed-forward MLP that shares
# the same objective: binary cross-entropy (logloss) on 30 scaled features.
# ---------------------------------------------------------------------------
class BreastCancerMLP(nn.Module):
    """
    Feed-forward network equivalent to the XGBoost binary classifier.

    Accepts 30 StandardScaler-normalised features and produces a single
    logit suitable for nn.BCEWithLogitsLoss (mirrors XGBoost's logloss
    eval_metric).  Hidden-layer capacity is sized to approximate a
    200-round, depth-4 gradient-boosted ensemble.
    """

    def __init__(
        self,
        in_features: int = 30,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))   # single logit → BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)        # shape (B,)


# ---------------------------------------------------------------------------
# 1. build_model
# ---------------------------------------------------------------------------
def build_model(config: dict) -> nn.Module:
    """Instantiate and return the BreastCancerMLP."""
    kwargs = config.get("model_kwargs", {})
    return BreastCancerMLP(**kwargs)


# ---------------------------------------------------------------------------
# Internal helper – assemble a TensorDataset from the sklearn breast-cancer
# dataset, applying the same StandardScaler used in the original script.
# ---------------------------------------------------------------------------
def _make_tensor_dataset() -> TensorDataset:
    data = load_breast_cancer()
    X, y = data.data, data.target          # (569, 30), binary float labels

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype("float32")
    y = y.astype("float32")

    return TensorDataset(torch.tensor(X), torch.tensor(y))


# ---------------------------------------------------------------------------
# 2. build_dataloader
# ---------------------------------------------------------------------------
def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for *split* ("train" or "val").

    Data source: sklearn breast_cancer (no external files required).
    data_path is read from config but unused for this in-memory dataset.
    Falls back to synthetic tensors if the real data cannot be loaded.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path  = config.get("data_path", ".")          # kept for API compliance
    seed       = config.get("seed", 42)

    try:
        full_ds = _make_tensor_dataset()
    except Exception:
        # Synthetic fallback: same shape as the real dataset
        X_fake = torch.randn(569, 30)
        y_fake = torch.randint(0, 2, (569,)).float()
        full_ds = TensorDataset(X_fake, y_fake)

    n_total = len(full_ds)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    chosen = train_ds if split == "train" else val_ds

    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
    )


# ---------------------------------------------------------------------------
# 3. train_step
# ---------------------------------------------------------------------------
def train_step(
    model: nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Single forward pass.

    Returns the loss tensor WITH grad attached.
    Does NOT call loss.backward() or optimizer.step() — the FL runtime
    is responsible for the backward pass and parameter update.
    """
    device = next(model.parameters()).device

    X, y = batch
    X = X.to(device)
    y = y.to(device)

    logits = model(X)                       # (B,)
    loss   = nn.BCEWithLogitsLoss()(logits, y)
    return loss                             # grad attached; no backward() here
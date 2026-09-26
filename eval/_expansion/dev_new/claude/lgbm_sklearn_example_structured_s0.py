# coding: utf-8
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


# ── preserved custom eval metrics from original script ──────────────────────
def rmsle(y_true, y_pred):
    return "RMSLE", np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False


def rae(y_true, y_pred):
    return "RAE", np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(np.mean(y_true) - y_true)), False


# ── PyTorch equivalent of lgb.LGBMRegressor ─────────────────────────────────
# LightGBM (num_leaves=31, lr=0.05, n_estimators=20) is replaced by a
# compact MLP that provides comparable capacity on tabular regression tasks.
class LGBMEquivalentRegressor(nn.Module):
    """
    MLP regression model equivalent to lgb.LGBMRegressor for tabular data.
    Mirrors the original script's num_leaves / depth budget with three
    hidden layers [128, 64, 32] and batch-norm + dropout regularisation.
    """

    def __init__(self, input_dim: int = 28, hidden_dims=None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))  # scalar regression output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Dataset wrapper ──────────────────────────────────────────────────────────
class TabularRegressionDataset(Dataset):
    """Wraps pandas DataFrame features + target as a PyTorch Dataset."""

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ── FL interface ─────────────────────────────────────────────────────────────
def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the LGBMEquivalentRegressor.

    Relevant config keys (all optional):
        config["model_kwargs"]["input_dim"]    – number of input features (default 28)
        config["model_kwargs"]["hidden_dims"]  – list of hidden layer widths
        config["model_kwargs"]["dropout"]      – dropout probability
    """
    model_kwargs = config.get("model_kwargs", {})
    model = LGBMEquivalentRegressor(**model_kwargs)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split.

    Reads:
        config["data_path"]               – directory containing regression.train
        config["local"]["batch_size"]     – mini-batch size (default 16)
        config["allow_synthetic_data"]    – must be True to allow synthetic fallback

    The full training file is split 80 / 20 into train and val subsets using
    random_split; the same deterministic partition is used for both calls.
    """
    batch_size: int = config.get("local", {}).get("batch_size", 16)
    data_path = Path(config.get("data_path", "."))
    train_file = data_path / "regression.train"

    if not train_file.exists():
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Regression training data not found at '{train_file}'. "
                "Provide the correct 'data_path' in config, or set "
                "config['allow_synthetic_data'] = True to use synthetic data."
            )
        # ── synthetic data fallback (gated) ─────────────────────────────────
        n_samples = 500
        input_dim: int = config.get("model_kwargs", {}).get("input_dim", 28)
        X_synth = torch.randn(n_samples, input_dim)
        y_synth = torch.randn(n_samples)
        full_ds = torch.utils.data.TensorDataset(X_synth, y_synth)
    else:
        df = pd.read_csv(str(train_file), header=None, sep="\t")
        y_all = df[0]
        X_all = df.drop(0, axis=1)
        full_ds = TabularRegressionDataset(X_all, y_all)

    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    # Reproducible split – seed fixed so train/val partitions are consistent.
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    if split == "train":
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    elif split == "val":
        return DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'train' or 'val'.")


def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Execute ONE forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT call either.

    Loss: MSE (matches the original script's RMSE / L2 regression objective).
    """
    device = next(model.parameters()).device

    X, y = batch
    X = X.to(device)
    y = y.to(device)

    preds = model(X)                    # shape: (B,)
    loss = F.mse_loss(preds, y)         # scalar, grad attached
    return loss                         # caller handles backward + step
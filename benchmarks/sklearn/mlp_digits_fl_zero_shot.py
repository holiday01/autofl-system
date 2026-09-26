"""
Federated Learning client module — MLP on the Digits dataset.

Exposes:
    build_model(config)                        -> torch.nn.Module
    build_dataloader(config, split)            -> torch.utils.data.DataLoader
    train_step(model, batch, optimizer, config) -> float  (loss)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Module-level cache so train/test splits share the same fitted scaler.
# ---------------------------------------------------------------------------
_data_cache: dict = {}


def _load_and_preprocess(config: dict) -> dict:
    key = (config.get("test_size", 0.2), config.get("seed", 42))
    if key not in _data_cache:
        digits = load_digits()
        X, y = digits.data, digits.target  # (1797, 64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=key[0],
            random_state=key[1],
            stratify=y,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        _data_cache[key] = {
            "train": (X_train.astype(np.float32), y_train.astype(np.int64)),
            "test":  (X_test.astype(np.float32),  y_test.astype(np.int64)),
        }
    return _data_cache[key]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_sizes: list[int], num_classes: int):
        super().__init__()
        dims = [input_dim] + list(hidden_layer_sizes)
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_model(config: dict) -> nn.Module:
    """
    config keys (all optional):
        hidden_layer_sizes  list[int]   default [128, 64]
        input_dim           int         default 64   (Digits feature count)
        num_classes         int         default 10
    """
    return _MLP(
        input_dim=config.get("input_dim", 64),
        hidden_layer_sizes=config.get("hidden_layer_sizes", [128, 64]),
        num_classes=config.get("num_classes", 10),
    )


def build_dataloader(config: dict, split: str) -> DataLoader:
    """
    config keys (all optional):
        test_size   float   default 0.2
        seed        int     default 42
        batch_size  int     default 32

    split: "train" | "test"
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    splits = _load_and_preprocess(config)
    X, y = splits[split]

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=(split == "train"),
    )


def train_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> float:
    """
    Performs one gradient-update step.

    Returns:
        Cross-entropy loss value (float) for this batch.
    """
    model.train()
    X, y = batch
    optimizer.zero_grad()
    loss = F.cross_entropy(model(X), y)
    loss.backward()
    optimizer.step()
    return loss.item()
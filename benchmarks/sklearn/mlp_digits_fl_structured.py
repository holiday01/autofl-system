import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# ---------------------------------------------------------------------------
# Model — PyTorch equivalent of sklearn MLPClassifier(hidden_layer_sizes=(...))
# Preserves: ReLU activations between every hidden layer, linear output layer.
# ---------------------------------------------------------------------------

class MLPClassifierNet(nn.Module):
    """Two-hidden-layer MLP for 10-class digit classification.

    Mirrors sklearn's MLPClassifier with ReLU activations, reproducing the
    original default architecture of hidden_layer_sizes=(128, 64).
    """

    def __init__(
        self,
        input_size: int = 64,
        hidden_layer_sizes: tuple = (128, 64),
        num_classes: int = 10,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_size
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the MLP model.

    Reads constructor arguments from config.get("model_kwargs", {}).
    Supported keys:
      - input_size        (int,  default 64)   — flattened 8×8 digit pixels
      - hidden_layer_sizes (list, default [128, 64])
      - num_classes       (int,  default 10)
    """
    kwargs = config.get("model_kwargs", {})
    input_size = int(kwargs.get("input_size", 64))
    hidden_layer_sizes = tuple(int(h) for h in kwargs.get("hidden_layer_sizes", [128, 64]))
    num_classes = int(kwargs.get("num_classes", 10))
    return MLPClassifierNet(
        input_size=input_size,
        hidden_layer_sizes=hidden_layer_sizes,
        num_classes=num_classes,
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    Data source: sklearn.datasets.load_digits (no external files required).
    data_path from config is accepted but unused because the dataset is
    embedded in scikit-learn.

    Falls back to synthetic tensors (torch.randn / torch.randint) if the
    sklearn dataset is unavailable for any reason.

    Config keys read:
      config["local"]["batch_size"]  — default 16
      config["data_path"]            — default "." (reserved; not used here)
      config["val_fraction"]         — default 0.2
      config["seed"]                 — default 42
    """
    batch_size: int = config.get("local", {}).get("batch_size", 16)
    _data_path: str = config.get("data_path", ".")          # accepted, not used
    val_fraction: float = float(config.get("val_fraction", 0.2))
    seed: int = int(config.get("seed", 42))

    # ── Real dataset ────────────────────────────────────────────────────────
    try:
        digits = load_digits()
        X, y = digits.data, digits.target                    # (1797, 64), 0-9

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype("float32")

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        full_dataset: TensorDataset = TensorDataset(X_tensor, y_tensor)

    # ── Synthetic fallback ───────────────────────────────────────────────────
    except Exception:
        n_samples = 1797
        X_tensor = torch.randn(n_samples, 64)
        y_tensor = torch.randint(0, 10, (n_samples,))
        full_dataset = TensorDataset(X_tensor, y_tensor)

    # ── Train / val split ────────────────────────────────────────────────────
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    chosen = train_subset if split == "train" else val_subset
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=False,
    )


def train_step(
    model: nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.
    """
    device = next(model.parameters()).device

    x, y = batch
    x = x.to(device)
    y = y.to(device)

    logits = model(x)                               # (B, num_classes)
    loss = F.cross_entropy(logits, y)               # scalar, grad attached
    return loss
"""
Auto-generated FL client module.
Original script: autofl/examples/digits_mlp_train.py

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ── Original source (unchanged) ────────────────────────────────────────
"""
MLP classifier on the Digits dataset (sklearn.neural_network.MLPClassifier).

Trains a two-hidden-layer MLP to recognise hand-written digits (0-9).
No external data files needed — loads via sklearn.datasets.
"""
import argparse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def main():
    parser = argparse.ArgumentParser(description="MLP on Digits (sklearn)")
    parser.add_argument("--hidden-layer-sizes", type=int, nargs="+", default=[128, 64],
                        help="Sizes of hidden layers (default: 128 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max training iterations (default: 200)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target  # (1797, 64), labels 0-9

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Normalise features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layer_sizes),
        learning_rate_init=args.lr,
        max_iter=args.epochs,
        random_state=args.seed,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print(metrics.classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()


# ── FL Interface ────────────────────────────────────────────────────────

class DigitsDataset(Dataset):
    """Wraps sklearn load_digits with optional StandardScaler normalisation."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DigitsMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_layer_sizes=(128, 64), num_classes=10, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layer_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(config: dict) -> nn.Module:
    """Instantiate the model. Override __init__ kwargs via config['model_kwargs']."""
    kwargs = config.get("model_kwargs", {})
    return DigitsMLP(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for the requested split.
    Loads sklearn digits, applies StandardScaler fit on training portion,
    then splits into train/val by config['val_ratio'].
    Client-local batch_size and num_workers are read from config['local'].
    """
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 0))
    pin_memory  = local.get("pin_memory", True)

    seed      = config.get("seed", 42)
    val_ratio = config.get("val_ratio", 0.1)
    normalize = config.get("normalize", True)

    digits = load_digits()
    X, y = digits.data.astype(np.float32), digits.target

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    full_dataset = DigitsDataset(X, y)
    n_val   = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
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
    Perform one forward (+ optionally backward) step.
    If optimizer is None (preflight forward-only check), skip backward.
    """
    local   = config.get("local", {})
    use_amp = local.get("use_amp", False)
    device  = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs  = batch.get("input", batch.get("x"))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    with torch.autocast(device_type=device.type if hasattr(device, "type") else str(device),
                        enabled=use_amp):
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

    if optimizer is not None:
        if use_amp:
            scaler = getattr(train_step, "_scaler", None)
            if scaler is None:
                train_step._scaler = torch.cuda.amp.GradScaler()
                scaler = train_step._scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    return loss.detach()
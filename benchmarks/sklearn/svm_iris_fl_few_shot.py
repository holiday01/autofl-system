"""
Auto-generated FL client module.
Original script: svm_iris.py

Exposes:
  build_model(config)               -> sklearn SVC
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn import svm, metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Original source (unchanged) ────────────────────────────────────────
"""
SVM classifier on the Iris dataset (sklearn).

Trains a Support Vector Machine with an RBF kernel to classify the three
Iris species.  No external data files needed — loads via sklearn.datasets.
"""
import argparse
from sklearn import svm, metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description="SVM on Iris (sklearn)")
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularisation (default: 1.0)")
    parser.add_argument("--kernel", type=str, default="rbf", help="SVM kernel (default: rbf)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Normalise features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    clf = svm.SVC(C=args.C, kernel=args.kernel, random_state=args.seed, probability=True)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))


if __name__ == "__main__":
    main()


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict):
    """Instantiate the SVM classifier. Override kwargs via config['model_kwargs']."""
    kwargs = config.get("model_kwargs", {})
    return svm.SVC(
        C=kwargs.get("C", config.get("C", 1.0)),
        kernel=kwargs.get("kernel", config.get("kernel", "rbf")),
        random_state=config.get("seed", 42),
        probability=True,
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Load Iris data, apply StandardScaler, and return a DataLoader backed by TensorDataset.
    config keys: 'test_size', 'val_ratio', 'seed', 'local.batch_size', 'local.num_workers'
    """
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 0))

    seed      = config.get("seed", 42)
    test_size = config.get("test_size", 0.2)
    val_ratio = config.get("val_ratio", 0.1)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)

    if split == "test":
        ds = TensorDataset(
            torch.from_numpy(X_test_scaled),
            torch.from_numpy(y_test_raw.astype(np.int64)),
        )
    else:
        full_ds = TensorDataset(
            torch.from_numpy(X_train_scaled),
            torch.from_numpy(y_train_raw.astype(np.int64)),
        )
        n_val   = max(1, int(len(full_ds) * val_ratio))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        ds = train_ds if split == "train" else val_ds

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
    )


def train_step(
    model,
    batch: tuple | list,
    optimizer,  # unused — sklearn SVC uses internal optimisation; accepted for interface parity
    config: dict,
) -> torch.Tensor:
    """
    Fit the SVM on one batch and return mean hinge loss as a scalar tensor.
    For multi-class OvR, hinge loss is computed against the true-class margin.
    """
    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        inputs  = batch.get("input", batch.get("x"))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    X_np = inputs.cpu().numpy()
    y_np = targets.cpu().numpy()

    model.fit(X_np, y_np)

    decision = model.decision_function(X_np)
    if decision.ndim == 1:
        # Binary: signed margin per sample
        y_signed = np.where(y_np == model.classes_[1], 1.0, -1.0)
        hinge = np.maximum(0.0, 1.0 - y_signed * decision).mean()
    else:
        # Multi-class OvR: margin of the ground-truth class column
        class_to_idx = {c: i for i, c in enumerate(model.classes_)}
        true_idx     = np.array([class_to_idx[c] for c in y_np])
        true_margins = decision[np.arange(len(y_np)), true_idx]
        hinge        = np.maximum(0.0, 1.0 - true_margins).mean()

    return torch.tensor(float(hinge), dtype=torch.float32)
"""
Auto-generated FL client module.
Original script: xgboost_breast_cancer_train.py

Exposes:
  build_model(config)               -> xgb.XGBClassifier
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor
"""
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# ── Original source (unchanged) ────────────────────────────────────────
"""
XGBoost binary classifier on the breast cancer dataset.

Trains an XGBoost gradient-boosted tree ensemble to distinguish malignant
from benign breast tumours.  No external data files needed — loads via
sklearn.datasets.
"""


def main():
    parser = argparse.ArgumentParser(description="XGBoost on breast_cancer (sklearn dataset)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of boosting rounds (default: 200)")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Maximum tree depth (default: 4)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate / eta (default: 0.1)")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="Row sub-sample ratio (default: 0.8)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target  # (569, 30), binary labels

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Normalise features (optional but consistent with sklearn scripts)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=args.seed,
        verbosity=1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10,
    )

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"\nTest accuracy : {acc:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print(metrics.classification_report(y_test, y_pred, target_names=data.target_names))


if __name__ == "__main__":
    main()


# ── FL Interface ────────────────────────────────────────────────────────

class BreastCancerDataset(Dataset):
    """In-memory breast cancer dataset with optional standardisation and split."""

    def __init__(self, split: str = "train", test_size: float = 0.2,
                 seed: int = 42, scale: bool = True):
        data = load_breast_cancer()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
            X_test = scaler.transform(X_test).astype(np.float32)

        self.X = X_train if split == "train" else X_test
        self.y = y_train if split == "train" else y_test

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def build_model(config: dict) -> xgb.XGBClassifier:
    """Instantiate the XGBClassifier. Override defaults via config['model_kwargs']."""
    kwargs = config.get("model_kwargs", {})
    return xgb.XGBClassifier(
        n_estimators=kwargs.get("n_estimators", 200),
        max_depth=kwargs.get("max_depth", 4),
        learning_rate=kwargs.get("learning_rate", 0.1),
        subsample=kwargs.get("subsample", 0.8),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=config.get("seed", 42),
        verbosity=kwargs.get("verbosity", 0),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for the requested split.
    Reads test_size, seed, and scale from config['dataset_kwargs'].
    Client-local batch_size and num_workers are read from config['local'].
    """
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 0))
    pin_memory  = local.get("pin_memory", True)

    dataset_kwargs = config.get("dataset_kwargs", {})
    ds = BreastCancerDataset(
        split=split,
        test_size=dataset_kwargs.get("test_size", 0.2),
        seed=config.get("seed", 42),
        scale=dataset_kwargs.get("scale", True),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: xgb.XGBClassifier,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Fit the XGBClassifier on one batch and return the log-loss as a scalar tensor.
    `optimizer` is ignored — XGBoost manages its own optimisation internally.
    The model is re-fitted on the provided batch; pass the full local dataset
    as a single batch (drop_last=False, batch_size=len(dataset)) for standard
    federated tree training.
    """
    if isinstance(batch, (list, tuple)):
        X, y = batch[0], batch[1]
    elif isinstance(batch, dict):
        X = batch.get("input", batch.get("x"))
        y = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

    model.fit(X_np, y_np)

    y_prob = model.predict_proba(X_np)[:, 1]
    loss_val = float(metrics.log_loss(y_np, y_prob))
    return torch.tensor(loss_val, dtype=torch.float32)
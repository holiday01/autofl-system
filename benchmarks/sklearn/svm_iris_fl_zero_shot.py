"""
Federated Learning client module — SVM on Iris (sklearn).

build_model     : constructs an SGDClassifier(loss="hinge"), a linear SVM
                  approximation that supports partial_fit for incremental FL updates.
build_dataloader: returns a list of (X_batch, y_batch) numpy tuples for the
                  requested split; scaling is fitted on local train data only.
train_step      : calls partial_fit on one batch; optimizer=None is accepted for
                  API compatibility (sklearn optimises internally).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Batch = Tuple[np.ndarray, np.ndarray]
Dataloader = List[Batch]


def build_model(config: Dict[str, Any]) -> SGDClassifier:
    C = config.get("C", 1.0)
    n_samples = config.get("n_train_samples", 120)
    # SGDClassifier alpha is the L2 penalty: alpha ≈ 1 / (C * n_samples)
    alpha = 1.0 / (C * n_samples)
    return SGDClassifier(
        loss=config.get("loss", "hinge"),
        alpha=alpha,
        max_iter=1,
        tol=None,
        warm_start=True,
        random_state=config.get("seed", 42),
    )


def build_dataloader(config: Dict[str, Any], split: str) -> Dataloader:
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("seed", 42),
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_split, y_split = (X_train, y_train) if split == "train" else (X_test, y_test)

    batch_size = config.get("batch_size", 32)
    return [
        (X_split[i : i + batch_size], y_split[i : i + batch_size])
        for i in range(0, len(X_split), batch_size)
    ]


def train_step(
    model: SGDClassifier,
    batch: Batch,
    optimizer: Optional[Any],
    config: Dict[str, Any],
) -> Dict[str, float]:
    X_batch, y_batch = batch
    all_classes = np.arange(config.get("num_classes", 3))
    model.partial_fit(X_batch, y_batch, classes=all_classes)
    loss = float(np.mean(model.predict(X_batch) != y_batch))
    return {"loss": loss}
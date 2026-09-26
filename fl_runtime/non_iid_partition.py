"""
Dirichlet partitioner for non-IID FL experiments.

Given a dataset and the number of clients/classes, this draws each client's
class proportions from a Dirichlet(alpha) distribution. Smaller alpha produces
more skewed partitions (some clients see few classes); larger alpha produces
near-IID partitions.

Reference: Hsu, Qi, Brown (2019) "Measuring the Effects of Non-Identical Data
Distribution for Federated Visual Classification."
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int = 0,
    min_per_client: int = 1,
) -> list[list[int]]:
    """Partition indices among clients using a Dirichlet class distribution.

    Args:
        labels: integer class labels for every dataset example (length N).
        num_clients: number of clients to split among.
        num_classes: total number of classes.
        alpha: Dirichlet concentration. Smaller -> more skewed.
        seed: RNG seed.
        min_per_client: ensure every client has at least this many samples.

    Returns:
        List of length num_clients; each element is a sorted list of indices.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    # Group indices by class
    class_to_indices: list[np.ndarray] = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        class_to_indices.append(idx)

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    # For each class, draw a Dirichlet(alpha) over clients and split that
    # class's indices accordingly.
    for c in range(num_classes):
        idx = class_to_indices[c]
        if len(idx) == 0:
            continue
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # Convert proportions into integer split points
        split_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, split_points)
        for client_id, chunk in enumerate(splits):
            client_indices[client_id].extend(int(i) for i in chunk)

    # Guarantee no client is empty by reassigning a few samples from the
    # largest client if needed.
    sizes = [len(s) for s in client_indices]
    for cid, sz in enumerate(sizes):
        if sz < min_per_client:
            donor = int(np.argmax([len(s) for s in client_indices]))
            need = min_per_client - sz
            moved = client_indices[donor][:need]
            client_indices[donor] = client_indices[donor][need:]
            client_indices[cid].extend(moved)

    for s in client_indices:
        s.sort()

    return client_indices


def iid_partition(
    num_samples: int,
    num_clients: int,
    seed: int = 0,
) -> list[list[int]]:
    """Random shuffle then equal split (IID baseline)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(num_samples)
    rng.shuffle(idx)
    chunks = np.array_split(idx, num_clients)
    return [sorted(int(i) for i in chunk) for chunk in chunks]


def partition_summary(
    client_indices: list[list[int]],
    labels: Sequence[int],
    num_classes: int,
) -> list[dict]:
    """Return per-client size + class histogram for logging."""
    labels = np.asarray(labels)
    summary = []
    for cid, idx in enumerate(client_indices):
        if len(idx) == 0:
            hist = [0] * num_classes
        else:
            hist = np.bincount(labels[idx], minlength=num_classes).tolist()
        summary.append({
            "client_id": cid,
            "size": len(idx),
            "class_hist": hist,
        })
    return summary

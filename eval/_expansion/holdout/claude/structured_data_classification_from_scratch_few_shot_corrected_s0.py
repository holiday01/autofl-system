"""
Auto-generated FL client module.
Original script: heart_disease_classification.py (Keras structured data example)

Exposes:
  build_model(config)                    -> nn.Module
  build_dataloader(config, split)        -> DataLoader
  train_step(model, batch, opt, config)  -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

NUMERIC_FEATURE_NAMES = ["age", "trestbps", "thalach", "oldpeak", "slope", "chol"]
CATEGORICAL_FEATURE_NAMES = ["sex", "cp", "fbs", "restecg", "exang", "ca", "thal"]
TARGET_FEATURE_NAME = "target"
DEFAULT_DATA_URL = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"


class HeartDiseaseDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_means: dict,
        num_stds: dict,
        cat_vocabs: dict,
    ):
        df = dataframe.copy()
        parts = []

        for col in NUMERIC_FEATURE_NAMES:
            if col in df.columns:
                vals = (df[col].values.astype(np.float32) - num_means[col]) / num_stds[col]
                parts.append(vals.reshape(-1, 1))

        for col in CATEGORICAL_FEATURE_NAMES:
            if col in df.columns:
                vocab = cat_vocabs[col]
                vocab_map = {v: i for i, v in enumerate(vocab)}
                ohe = np.zeros((len(df), len(vocab)), dtype=np.float32)
                for i, v in enumerate(df[col].values):
                    idx = vocab_map.get(v)
                    if idx is not None:
                        ohe[i, idx] = 1.0
                parts.append(ohe)

        self.X = torch.from_numpy(np.concatenate(parts, axis=1))
        self.y = torch.from_numpy(df[TARGET_FEATURE_NAME].values.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HeartDiseaseClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── helpers ─────────────────────────────────────────────────────────────

def _load_dataframe(config: dict) -> pd.DataFrame:
    data_path = config.get("data_path", DEFAULT_DATA_URL)
    return pd.read_csv(data_path)


def _split_dataframe(df: pd.DataFrame, config: dict):
    val_frac = config.get("val_ratio", 0.2)
    val_df = df.sample(frac=val_frac, random_state=config.get("seed", 1337))
    train_df = df.drop(val_df.index)
    return train_df, val_df


def _build_preprocessing(train_df: pd.DataFrame, full_df: pd.DataFrame):
    num_means = {col: float(train_df[col].mean()) for col in NUMERIC_FEATURE_NAMES if col in train_df.columns}
    num_stds  = {col: float(train_df[col].std()) + 1e-8 for col in NUMERIC_FEATURE_NAMES if col in train_df.columns}
    # Vocab from full dataframe so validation split has no unknown values
    cat_vocabs = {
        col: sorted(full_df[col].unique().tolist())
        for col in CATEGORICAL_FEATURE_NAMES if col in full_df.columns
    }
    return num_means, num_stds, cat_vocabs


def _input_dim(cat_vocabs: dict) -> int:
    return len(NUMERIC_FEATURE_NAMES) + sum(len(v) for v in cat_vocabs.values())


# ── FL Interface ─────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    if "input_dim" in config.get("model_kwargs", {}):
        input_dim = config["model_kwargs"]["input_dim"]
    else:
        df = _load_dataframe(config)
        train_df, _ = _split_dataframe(df, config)
        _, _, cat_vocabs = _build_preprocessing(train_df, df)
        input_dim = _input_dim(cat_vocabs)
    kwargs = {k: v for k, v in config.get("model_kwargs", {}).items() if k != "input_dim"}
    return HeartDiseaseClassifier(input_dim=input_dim, **kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 0))
    pin_memory  = local.get("pin_memory", True)

    df = _load_dataframe(config)
    train_df, val_df = _split_dataframe(df, config)
    num_means, num_stds, cat_vocabs = _build_preprocessing(train_df, df)

    source_df = train_df if split == "train" else val_df
    dataset = HeartDiseaseDataset(source_df, num_means, num_stds, cat_vocabs)

    return DataLoader(
        dataset,
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
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
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

    outputs = model(inputs)
    loss = nn.BCELoss()(outputs, targets)
    return loss
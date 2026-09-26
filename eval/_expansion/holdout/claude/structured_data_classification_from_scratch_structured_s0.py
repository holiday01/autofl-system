import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# ──────────────────────────────────────────────────────────────────────────────
# Dataset metadata (preserved from original script)
# ──────────────────────────────────────────────────────────────────────────────

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
TARGET_FEATURE_NAME = "target"
NUMERIC_FEATURE_NAMES = ["age", "trestbps", "thalach", "oldpeak", "slope", "chol"]
CATEGORICAL_FEATURE_NAMES = [
    c for c in COLUMN_NAMES
    if c not in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME]
]  # ["sex", "cp", "fbs", "restecg", "exang", "ca", "thal"]

FILE_URL = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_dataframe(data_path: str) -> pd.DataFrame:
    """Load heart.csv from a local directory; fall back to the remote URL."""
    local_path = os.path.join(data_path, "heart.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    return pd.read_csv(FILE_URL)


def _build_cat_vocab(df: pd.DataFrame) -> dict:
    """
    Build sorted vocabulary lists for every categorical feature.
    Integer-typed (int64) columns keep integer values; all other columns
    are cast to str — mirrors the original CATEGORICAL_FEATURES_WITH_VOCABULARY.
    """
    return {
        feat: sorted(
            [v if df[feat].dtype == "int64" else str(v)
             for v in df[feat].unique()]
        )
        for feat in CATEGORICAL_FEATURE_NAMES
    }


class HeartDiseaseDataset(Dataset):
    """
    PyTorch Dataset for the Cleveland Heart Disease CSV.

    Numerical features  →  z-score normalised (statistics from `stats_df`).
    Categorical features →  one-hot encoded (vocabulary from `cat_vocab`).

    Parameters
    ----------
    df        : DataFrame whose rows become dataset items.
    stats_df  : DataFrame used to compute normalisation mean / std
                (pass the training slice to avoid leakage when desired;
                 passing the full df is acceptable for the small dataset).
    cat_vocab : dict mapping feature name → sorted vocabulary list.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stats_df: pd.DataFrame,
        cat_vocab: dict,
    ) -> None:
        super().__init__()

        # ── Numerical features: z-score normalisation ─────────────────────
        means = stats_df[NUMERIC_FEATURE_NAMES].mean().values.astype(np.float32)
        stds  = stats_df[NUMERIC_FEATURE_NAMES].std(ddof=0).values.astype(np.float32)
        stds[stds == 0.0] = 1.0            # guard against constant columns

        num_mat = df[NUMERIC_FEATURE_NAMES].values.astype(np.float32)
        num_mat = (num_mat - means) / stds

        # ── Categorical features: one-hot encoding ─────────────────────────
        ohe_parts: list = []
        for feat in CATEGORICAL_FEATURE_NAMES:
            vocab   = cat_vocab[feat]
            val2idx = {v: i for i, v in enumerate(vocab)}
            col     = df[feat].values
            ohe     = np.zeros((len(df), len(vocab)), dtype=np.float32)
            for row_i, raw_v in enumerate(col):
                # Cast to the same Python type as vocabulary entries
                key = int(raw_v) if isinstance(vocab[0], int) else str(raw_v)
                idx = val2idx.get(key, -1)
                if idx >= 0:
                    ohe[row_i, idx] = 1.0
            ohe_parts.append(ohe)

        features = np.concatenate([num_mat] + ohe_parts, axis=1)
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(
            df[TARGET_FEATURE_NAME].values, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Model — exact PyTorch equivalent of the Keras Classifier layer:
#   Dense(32, relu) → Dropout(0.5) → Dense(1, sigmoid)
# ──────────────────────────────────────────────────────────────────────────────

class HeartDiseaseClassifier(nn.Module):
    """
    Binary classifier for structured heart-disease tabular data.

    Parameters
    ----------
    input_dim : int
        Total feature dimension after preprocessing.  Default 26 matches the
        Cleveland Heart Disease CSV:
          6 numerical (normalised)
        + 2 (sex) + 4 (cp) + 2 (fbs) + 3 (restecg) + 2 (exang)
        + 4 (ca) + 3 (thal)  = 20 one-hot dims
        = 26 total.
    """

    def __init__(self, input_dim: int = 26, **kwargs) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.dense_2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dense_1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.dense_2(x))


# ──────────────────────────────────────────────────────────────────────────────
# FL API
# ──────────────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the HeartDiseaseClassifier.

    Relevant config keys
    --------------------
    model_kwargs : dict
        Forwarded to HeartDiseaseClassifier.__init__.
        Use {"input_dim": N} when the feature dimension is not 26.
    """
    kwargs = config.get("model_kwargs", {})
    return HeartDiseaseClassifier(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    Relevant config keys
    --------------------
    local.batch_size     : int   batch size (default 16)
    data_path            : str   directory that may contain heart.csv (default ".")
    val_frac             : float fraction of data reserved for validation (default 0.2)
    seed                 : int   RNG seed for the train/val split (default 42)
    allow_synthetic_data : bool  allow synthetic fallback when real data is absent
                                 (default False — raises FileNotFoundError instead)
    model_kwargs.input_dim : int feature dim used for synthetic tensors (default 26)
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path  = config.get("data_path", ".")
    val_frac   = config.get("val_frac", 0.2)
    seed       = config.get("seed", 42)
    generator  = torch.Generator().manual_seed(seed)

    # ── Attempt to load the real dataset ──────────────────────────────────
    try:
        df = _load_dataframe(data_path)
    except Exception as exc:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Heart Disease CSV not found at '{os.path.join(data_path, 'heart.csv')}' "
                "and could not be fetched from the remote URL. "
                "Place heart.csv in data_path, ensure internet access, or set "
                "config['allow_synthetic_data'] = True to allow synthetic data."
            ) from exc

        # ── Synthetic fallback (only reached when explicitly opted-in) ─────
        input_dim = config.get("model_kwargs", {}).get("input_dim", 26)
        n         = 303
        X         = torch.randn(n, input_dim)
        y         = torch.randint(0, 2, (n,)).float()
        full_ds   = TensorDataset(X, y)
        n_val     = max(1, int(n * val_frac))
        n_train   = n - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val], generator=generator
        )
        subset = train_ds if split == "train" else val_ds
        return DataLoader(
            subset, batch_size=batch_size,
            shuffle=(split == "train"), drop_last=False
        )

    # ── Build a single HeartDiseaseDataset from the full dataframe,
    #    then use random_split to obtain train / val subsets ───────────────
    cat_vocab = _build_cat_vocab(df)
    full_ds   = HeartDiseaseDataset(df, stats_df=df, cat_vocab=cat_vocab)

    n       = len(full_ds)
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=generator
    )

    subset = train_ds if split == "train" else val_ds
    return DataLoader(
        subset, batch_size=batch_size,
        shuffle=(split == "train"), drop_last=False
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,          # accepted but intentionally NOT called here
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the scalar loss with grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must not call either.
    """
    device          = next(model.parameters()).device
    features, labels = batch
    features = features.to(device)
    labels   = labels.to(device).unsqueeze(1)   # (B,) → (B, 1)

    preds = model(features)                      # (B, 1), values ∈ [0, 1]
    loss  = F.binary_cross_entropy(preds, labels)
    return loss                                  # grad attached; no backward()
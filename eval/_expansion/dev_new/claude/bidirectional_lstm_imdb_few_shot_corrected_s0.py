"""
Auto-generated FL client module.
Original script: bidirectional_lstm_imdb.py

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class IMDBDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int = 20000, embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # return_sequences=True equivalent: keep all timesteps from lstm1
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # lstm2 consumes the full sequence output of lstm1
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        # No sigmoid here — BCEWithLogitsLoss in train_step handles it
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        out1, _ = self.lstm1(emb)
        out2, _ = self.lstm2(out1)
        last = out2[:, -1, :]
        return self.classifier(last)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return BiLSTMClassifier(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    max_features = config.get("max_features", 20000)
    maxlen       = config.get("maxlen", 200)
    val_ratio    = config.get("val_ratio", 0.1)
    seed         = config.get("seed", 42)

    try:
        import keras
        (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
            num_words=max_features
        )
        x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
        x_val   = keras.utils.pad_sequences(x_val,   maxlen=maxlen)
        ds = IMDBDataset(x_train, y_train) if split == "train" else IMDBDataset(x_val, y_val)
    except Exception:
        # synthetic fallback for testing without the IMDB download
        n = 500
        sequences = np.random.randint(1, max_features, (n, maxlen)).astype(np.int64)
        labels    = np.random.randint(0, 2, n).astype(np.float32)
        full_ds   = IMDBDataset(sequences, labels)
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
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
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
        inputs  = batch.get("input", batch.get("x", batch.get("tokens")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs).squeeze(-1)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs, targets)
    return loss
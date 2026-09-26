import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# ---------------------------------------------------------------------------
# Model – PyTorch equivalent of the Keras FCN from the original script
# ---------------------------------------------------------------------------

class _FCNBlock(nn.Module):
    """Conv1D → BatchNorm → ReLU block (matches one Keras conv block)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        # padding="same" keeps seq_len unchanged (stride=1 only, matches Keras default)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding="same", bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class FCNTimeSeries(nn.Module):
    """
    Fully Convolutional Network for timeseries classification.

    Faithfully reproduces the Keras architecture from the original script:
        Input  → Conv1D(64,3)+BN+ReLU
               → Conv1D(64,3)+BN+ReLU
               → Conv1D(64,3)+BN+ReLU
               → GlobalAveragePooling1D
               → Dense(num_classes)

    Input tensor shape (PyTorch channels-first): (batch, in_channels, seq_len)
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        filters: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.block1 = _FCNBlock(in_channels, filters, kernel_size)
        self.block2 = _FCNBlock(filters, filters, kernel_size)
        self.block3 = _FCNBlock(filters, filters, kernel_size)
        self.fc = nn.Linear(filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, seq_len)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=-1)   # Global Average Pooling over the time axis
        return self.fc(x)    # raw logits; CrossEntropyLoss handles softmax


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the FCN timeseries classifier.

    Recognised config.model_kwargs keys
    ------------------------------------
    num_classes  : int  – number of output classes          (default 2)
    in_channels  : int  – input channel count               (default 1)
    filters      : int  – Conv1D filter count in each block (default 64)
    kernel_size  : int  – convolution kernel size           (default 3)
    """
    kwargs = config.get("model_kwargs", {})
    return FCNTimeSeries(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split.

    Data layout expected on disk
    ----------------------------
    <data_path>/FordA_TRAIN.tsv – TSV where column 0 is the label {-1, 1}
                                   and columns 1‥500 are the timeseries values.

    The full training file is loaded and split 80 / 20 into train / val subsets
    via random_split (seed 42 for reproducibility).

    Synthetic fallback
    ------------------
    Activated only when config['allow_synthetic_data'] is True AND the TSV is
    absent.  If the file is missing and the flag is False, a FileNotFoundError
    is raised immediately – no silent training on fake data.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path  = config.get("data_path", ".")
    train_file = os.path.join(data_path, "FordA_TRAIN.tsv")

    # ------------------------------------------------------------------
    # Build the full dataset (real data or synthetic fallback)
    # ------------------------------------------------------------------
    if not os.path.isfile(train_file):
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"FordA training file not found at '{train_file}'. "
                "Place FordA_TRAIN.tsv under config['data_path'] or set "
                "config['allow_synthetic_data']=True to run on synthetic data."
            )
        # Synthetic fallback: shape mirrors the real dataset
        n_synthetic = 500
        seq_len     = 500
        x_syn = torch.randn(n_synthetic, 1, seq_len)
        y_syn = torch.randint(0, 2, (n_synthetic,))
        full_dataset = TensorDataset(x_syn, y_syn)
    else:
        raw  = np.loadtxt(train_file, delimiter="\t")
        y_np = raw[:, 0].astype(int)
        x_np = raw[:, 1:]

        # Map labels {-1, 1} → {0, 1} (matches original script)
        y_np[y_np == -1] = 0

        # Reshape to channels-first: (N, 1, seq_len) for PyTorch Conv1d
        x_np = x_np.reshape(x_np.shape[0], 1, x_np.shape[1])

        x_t = torch.tensor(x_np, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.long)
        full_dataset = TensorDataset(x_t, y_t)

    # ------------------------------------------------------------------
    # 80 / 20 random split (same source dataset for both splits)
    # ------------------------------------------------------------------
    n_total = len(full_dataset)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    chosen  = train_subset if split == "train" else val_subset
    shuffle = split == "train"
    return DataLoader(chosen, batch_size=batch_size, shuffle=shuffle)


def train_step(
    model: nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """
    One forward pass.  Returns the loss tensor WITH gradients attached.

    The FL runtime is responsible for loss.backward() and optimizer.step();
    this function must NOT call either.
    """
    device = next(model.parameters()).device

    x, y = batch
    x = x.to(device)   # (batch, 1, seq_len)
    y = y.to(device)   # (batch,)  long

    logits = model(x)                        # (batch, num_classes)
    loss   = F.cross_entropy(logits, y)      # scalar, grad attached
    return loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# ---------------------------------------------------------------------------
# Model — Keras Sequential converted to an equivalent torch.nn.Module.
#
# Original Keras architecture:
#   Input(28,28,1)
#   Conv2D(32, 3x3, relu)   → (26,26,32)
#   MaxPooling2D(2x2)        → (13,13,32)
#   Conv2D(64, 3x3, relu)   → (11,11,64)
#   MaxPooling2D(2x2)        → (5,5,64) = 1600 units
#   Flatten
#   Dropout(0.5)
#   Dense(10, softmax)
#
# PyTorch equivalent uses raw logits + F.cross_entropy, which is
# numerically identical to Keras softmax + categorical_crossentropy.
# ---------------------------------------------------------------------------

class MNISTConvNet(nn.Module):
    """PyTorch equivalent of the Keras simple MNIST convnet."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # (N,1,28,28)  → (N,32,26,26)
        self.pool1 = nn.MaxPool2d(2)                    #               → (N,32,13,13)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  #               → (N,64,11,11)
        self.pool2 = nn.MaxPool2d(2)                    #               → (N,64,5,5)
        self.flatten = nn.Flatten()                     #               → (N,1600)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1600, num_classes)          #               → (N,10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # raw logits; loss is computed in train_step


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the MNISTConvNet.

    Parameters read from config:
        config.get("model_kwargs", {})  →  passed as **kwargs to MNISTConvNet.
            Supported keys: num_classes (int, default 10),
                            dropout     (float, default 0.5).

    Returns:
        Un-trained MNISTConvNet instance.
    """
    kwargs = config.get("model_kwargs", {})
    return MNISTConvNet(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split of MNIST.

    Config keys used:
        config["data_path"]                → root directory for MNIST
                                             (downloaded there if absent, default ".")
        config["local"]["batch_size"]      → mini-batch size (default 16)
        config["allow_synthetic_data"]     → if True, fall back to random tensors
                                             when real data is unavailable (default False)

    Args:
        config: FL runtime configuration dict.
        split:  "train" or "val"  (80 / 20 random split from the full training set)

    Returns:
        DataLoader yielding (images, labels) tuples.
        images : FloatTensor  (B, 1, 28, 28), values in [0, 1]
        labels : LongTensor   (B,),            class indices 0-9
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic = config.get("allow_synthetic_data", False)

    # ------------------------------------------------------------------
    # Attempt to load real MNIST via torchvision
    # ------------------------------------------------------------------
    dataset = None
    load_error = None
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),   # uint8 [0,255] → float32 [0,1], adds channel dim
        ])
        dataset = datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transform,
        )
    except Exception as exc:
        load_error = exc

    # ------------------------------------------------------------------
    # Synthetic fallback — only if explicitly permitted
    # ------------------------------------------------------------------
    if dataset is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"Could not load the MNIST dataset from data_path='{data_path}': "
                f"{load_error}. "
                "Either make the data available at that path or set "
                "config['allow_synthetic_data'] = True to use random tensors instead."
            ) from load_error

        # Synthetic MNIST-shaped data: 1 000 random samples
        n_samples = 1000
        images = torch.randn(n_samples, 1, 28, 28)
        labels = torch.randint(0, 10, (n_samples,))
        dataset = TensorDataset(images, labels)

    # ------------------------------------------------------------------
    # 80 / 20 train / val split (reproducible)
    # ------------------------------------------------------------------
    total = len(dataset)
    val_size = max(1, int(0.2 * total))
    train_size = total - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_set if split == "train" else val_set
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=False,
        num_workers=0,
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,          # owned by the FL runtime — do NOT call .step() here
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the scalar loss tensor (grad attached).

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step().  This function must NOT do either.

    Args:
        model:     MNISTConvNet (or compatible nn.Module).
        batch:     (images, labels) tuple from the DataLoader produced by
                   build_dataloader.
        optimizer: Passed in by the FL runtime; not used here.
        config:    FL runtime configuration dict; not used here.

    Returns:
        Scalar CrossEntropyLoss tensor with gradient graph attached.
        Equivalent to Keras categorical_crossentropy on softmax outputs.
    """
    device = next(model.parameters()).device

    images, labels = batch
    images = images.to(device)          # (B, 1, 28, 28) float32
    labels = labels.to(device).long()   # (B,) int64

    logits = model(images)              # (B, num_classes) — raw logits
    loss = F.cross_entropy(logits, labels)
    # loss.backward() and optimizer.step() are intentionally omitted;
    # the FL runtime handles aggregation and weight updates.
    return loss
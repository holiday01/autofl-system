import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

# ---------------------------------------------------------------------------
# Model — converted from Keras Sequential to torch.nn.Module
#
# Original Keras layers (channels-last, NHWC):
#   Conv2D(32, 3x3, relu)  -> MaxPool2D(2x2)
#   Conv2D(64, 3x3, relu)  -> MaxPool2D(2x2)
#   Flatten -> Dropout(0.5) -> Dense(10, softmax)
#
# PyTorch uses channels-first (NCHW). Spatial walk-through:
#   28x28 -Conv(3)-> 26x26 -Pool(2)-> 13x13
#         -Conv(3)-> 11x11 -Pool(2)->  5x5
#   Flatten: 64 * 5 * 5 = 1600  -> Linear(1600, 10)
#   Softmax is folded into F.cross_entropy (log-sum-exp trick).
# ---------------------------------------------------------------------------

class MnistConvNet(nn.Module):
    """PyTorch equivalent of the original Keras MNIST convnet."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # (N,  1, 28, 28) -> (N, 32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(2),                    # (N, 32, 26, 26) -> (N, 32, 13, 13)
            nn.Conv2d(32, 64, kernel_size=3),  # (N, 32, 13, 13) -> (N, 64, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(2),                    # (N, 64, 11, 11) -> (N, 64,  5,  5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # 64 * 5 * 5 = 1600
            nn.Dropout(dropout),
            nn.Linear(64 * 5 * 5, num_classes), # 1600 -> 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the MnistConvNet.

    Recognised model_kwargs:
        num_classes (int, default 10)
        dropout     (float, default 0.5)
    """
    kwargs = config.get("model_kwargs", {})
    return MnistConvNet(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Data loading strategy
    ----------------------
    1. Try torchvision MNIST (downloaded to data_path if necessary).
    2. If unavailable (import error, network error, permission error, …)
       fall back to a fully synthetic dataset of the same shape so the FL
       round can still proceed.

    Config keys read
    ----------------
    config["local"]["batch_size"]  – default 16
    config["data_path"]            – root dir for MNIST download, default "."
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path  = config.get("data_path", ".")

    try:
        from torchvision import datasets, transforms

        # Mirror the Keras pre-processing:  float32, scale to [0, 1], shape (1,28,28)
        transform = transforms.Compose([transforms.ToTensor()])

        full_dataset = datasets.MNIST(
            root=data_path, train=True, download=True, transform=transform
        )
        n_val   = max(1, int(0.1 * len(full_dataset)))  # 10 % validation split
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    except Exception:
        # ------------------------------------------------------------------ #
        # Synthetic fallback – 1 000 random samples with the correct shapes  #
        # ------------------------------------------------------------------ #
        x_syn = torch.randn(1000, 1, 28, 28)           # float, (1, 28, 28)
        y_syn = torch.randint(0, 10, (1000,))           # integer class labels
        full_dataset = TensorDataset(x_syn, y_syn)
        n_val   = 100
        n_train = 900
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    dataset = train_ds if split == "train" else val_ds
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=False,
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,       # received from FL runtime; NOT called here
    config: dict,
) -> torch.Tensor:
    """
    Run ONE forward pass and return the scalar loss tensor WITH grad attached.

    Constraints (FL runtime contract)
    -----------------------------------
    - Do NOT call loss.backward().
    - Do NOT call optimizer.step() or optimizer.zero_grad().
    - Move input tensors to the device that holds the model parameters.

    Loss
    ----
    F.cross_entropy(logits, labels) — numerically equivalent to the original
    Keras categorical_crossentropy(softmax(logits), one_hot(labels)) but
    operates directly on integer class indices, which is what torchvision
    MNIST (and the synthetic fallback) provide.
    """
    device = next(model.parameters()).device

    images, labels = batch
    images = images.to(device)          # (N, 1, 28, 28), float
    labels = labels.to(device)          # (N,), long

    logits = model(images)              # (N, num_classes)
    loss   = F.cross_entropy(logits, labels)
    return loss                         # scalar tensor, grad_fn intact
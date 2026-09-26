"""
Auto-generated FL client module.
Original script: Simple MNIST convnet (fchollet, keras.io)

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT:
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class MNISTConvNet(nn.Module):
    """PyTorch port of the Keras MNIST convnet (fchollet).

    Input:  (N, 1, 28, 28)  — float32, values in [0, 1]
    Output: (N, num_classes) — raw logits (no softmax; CrossEntropyLoss handles it)

    Feature-map sizes after each block (default 28×28 input):
      Conv(3×3) → 26×26  |  MaxPool(2) → 13×13
      Conv(3×3) → 11×11  |  MaxPool(2) →  5×5
      Flatten  → 64 × 5 × 5 = 1600
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 5 * 5, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return MNISTConvNet(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  128))
    num_workers = local.get("num_workers", config.get("num_workers",  2))
    pin_memory  = local.get("pin_memory",  True)
    seed        = config.get("seed", 42)
    data_path   = config.get("data_path", ".")

    dataset = None

    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        is_train = split in ("train", "val")
        dataset = datasets.MNIST(
            root=data_path, train=is_train, download=False, transform=transform
        )
    except Exception:
        pass

    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST data not found at '{data_path}' (torchvision.datasets.MNIST "
                "failed or torchvision is not installed) and "
                "'allow_synthetic_data' is False."
            )
        num_classes = config.get("model_kwargs", {}).get("num_classes", 10)
        n = {"train": 500, "val": 100, "test": 100}.get(split, 500)
        x = torch.randn(n, 1, 28, 28)
        y = torch.randint(0, num_classes, (n,))
        return DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    if split in ("train", "val"):
        val_ratio = config.get("val_ratio", 0.1)
        n_val   = max(1, int(len(dataset) * val_ratio))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        dataset = train_ds if split == "train" else val_ds

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

    Targets must be integer class indices (torch.long), NOT one-hot vectors.
    torchvision.datasets.MNIST already provides integer labels.
    """
    device = next(model.parameters()).device
    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    return loss
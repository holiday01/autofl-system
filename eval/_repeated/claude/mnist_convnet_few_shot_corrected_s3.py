"""
Auto-generated FL client module.
Original script: keras_mnist_convnet.py

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
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class MNISTDataset(Dataset):
    """MNIST dataset wrapper; downloads via torchvision or falls back to keras."""

    def __init__(self, root: str = ".", train: bool = True):
        try:
            from torchvision import datasets, transforms
            ds = datasets.MNIST(
                root=root,
                train=train,
                download=True,
                transform=transforms.ToTensor(),
            )
            self.images = ds.data.unsqueeze(1).float() / 255.0
            self.labels = ds.targets
        except Exception:
            import keras
            (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
            x, y = (x_tr, y_tr) if train else (x_te, y_te)
            x = x.astype("float32") / 255.0
            self.images = torch.tensor(x).unsqueeze(1)
            self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class MNISTConvNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(64 * 5 * 5, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return MNISTConvNet(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 128))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    data_path = config.get("data_path", ".")
    is_train = split == "train"

    if split in ("train", "val"):
        full_dataset = MNISTDataset(root=data_path, train=True)
        val_ratio = config.get("val_ratio", 0.1)
        n_val = max(1, int(len(full_dataset) * val_ratio))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.get("seed", 42)),
        )
        ds = train_ds if split == "train" else val_ds
    else:
        ds = MNISTDataset(root=data_path, train=False)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
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
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss
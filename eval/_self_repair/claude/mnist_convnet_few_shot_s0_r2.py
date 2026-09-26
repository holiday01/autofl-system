"""
Auto-generated FL client module.
Original script: mnist_convnet_keras.py

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class MNISTDataset(Dataset):
    """MNIST images (HWC numpy, [0,1]) converted to NCHW float tensors."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.x = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MNISTConvNet(nn.Module):
    """PyTorch port of the Keras convnet: Conv→Pool→Conv→Pool→Flatten→Drop→Linear."""

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


def _load_mnist_numpy():
    """Return (x_train, y_train, x_test, y_test) as numpy arrays via torchvision or keras."""
    try:
        from torchvision import datasets, transforms
        _t = transforms.ToTensor()
        _tr = datasets.MNIST(root="/tmp/mnist", train=True,  download=True, transform=_t)
        _te = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=_t)
        xt = _tr.data.numpy().astype("float32")[:, :, :, None] / 255
        yt = _tr.targets.numpy()
        xe = _te.data.numpy().astype("float32")[:, :, :, None] / 255
        ye = _te.targets.numpy()
        return xt, yt, xe, ye
    except Exception:
        import keras as _keras
        (xt, yt), (xe, ye) = _keras.datasets.mnist.load_data()
        xt = np.expand_dims(xt.astype("float32") / 255, -1)
        xe = np.expand_dims(xe.astype("float32") / 255, -1)
        return xt, yt, xe, ye


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate the model. Override __init__ kwargs via config['model_kwargs']."""
    kwargs = config.get("model_kwargs", {})
    return MNISTConvNet(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for 'train', 'val', or 'test'.
    MNIST is loaded via torchvision (fallback: keras.datasets).
    Client-local batch_size and num_workers are read from config['local'].
    """
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 128))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    x_train, y_train, x_test, y_test = _load_mnist_numpy()

    if split == "test":
        ds      = MNISTDataset(x_test, y_test)
        shuffle = False
    else:
        val_ratio = config.get("val_ratio", 0.1)
        n_val     = max(1, int(len(x_train) * val_ratio))
        n_train   = len(x_train) - n_val
        full_ds   = MNISTDataset(x_train, y_train)
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.get("seed", 42)),
        )
        ds      = train_ds if split == "train" else val_ds
        shuffle = split == "train"

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
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
    Perform one forward pass and, when optimizer is provided, a backward pass + step.

    Returns the loss tensor WITH grad_fn when optimizer is None so that the caller
    can invoke .backward() externally (e.g. harness preflight checks).  Returns a
    detached scalar when optimizer is provided and the step has already been applied.
    """
    model.train()

    local       = config.get("local", {})
    use_amp     = local.get("use_amp", False) and torch.cuda.is_available()
    device      = next(model.parameters()).device
    device_type = device.type if hasattr(device, "type") else "cpu"

    if isinstance(batch, (list, tuple)):
        batch   = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    with torch.autocast(device_type=device_type, enabled=use_amp):
        outputs = model(inputs)
        loss    = nn.CrossEntropyLoss()(outputs, targets)

    if optimizer is None:
        # Return live loss so the caller can call .backward() for preflight checks.
        return loss

    optimizer.zero_grad()
    if use_amp:
        scaler = getattr(train_step, "_scaler", None)
        if scaler is None:
            train_step._scaler = torch.amp.GradScaler()
            scaler = train_step._scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.detach()
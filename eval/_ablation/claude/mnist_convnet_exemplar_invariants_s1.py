"""
Auto-generated FL client module.
Original script: Simple MNIST convnet (fchollet, 2015/06/19)

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

NUM_CLASSES = 10
# After Conv2d(1,32,3)->MaxPool2d(2)->Conv2d(32,64,3)->MaxPool2d(2) on 28x28: 64*5*5
_FLAT_DIM = 1600


class MNISTConvNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(_FLAT_DIM, num_classes),
        )

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
    val_ratio   = config.get("val_ratio", 0.1)
    seed        = config.get("seed", 42)
    data_root   = config.get("data_path", "./data")

    ds = None
    load_error = None
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        is_train_split = split in ("train", "val")
        full_ds = datasets.MNIST(
            root=data_root,
            train=is_train_split,
            download=config.get("download", True),
            transform=transform,
        )
        if split in ("train", "val"):
            n_val = max(1, int(len(full_ds) * val_ratio))
            n_train = len(full_ds) - n_val
            train_ds, val_ds = random_split(
                full_ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(seed),
            )
            ds = train_ds if split == "train" else val_ds
        else:
            ds = full_ds
    except Exception as exc:
        load_error = exc

    if ds is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST data unavailable at '{data_root}' ({load_error}) and "
                "allow_synthetic_data is False. Set config['allow_synthetic_data'] = True "
                "to use a synthetic fallback."
            )
        n = config.get("synthetic_n", 1000)
        X = torch.randn(n, 1, 28, 28)
        y = torch.randint(0, NUM_CLASSES, (n,))
        full_ds = TensorDataset(X, y)
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val
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
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    return loss
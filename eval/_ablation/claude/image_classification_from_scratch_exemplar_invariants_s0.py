"""
Auto-generated FL client module.
Original script: Keras image classification from scratch (Cats vs Dogs).

Exposes:
  build_model(config)                    -> nn.Module
  build_dataloader(config, split)        -> DataLoader
  train_step(model, batch, opt, config)  -> loss tensor (with grad_fn)

CONTRACT:
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image


class CatsDogsDataset(Dataset):
    """Binary image dataset from root/Cat/ and root/Dog/ subdirectories.
    Skips files that lack a JFIF header (corrupted JPEGs)."""

    CLASSES = ["Cat", "Dog"]

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        for label_idx, class_name in enumerate(self.CLASSES):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fpath = os.path.join(class_dir, fname)
                try:
                    with open(fpath, "rb") as f:
                        header = f.read(10)
                    if b"JFIF" not in header:
                        continue
                except OSError:
                    continue
                self.samples.append((fpath, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


class _SyntheticImageDataset(Dataset):
    def __init__(self, n: int, image_size: tuple):
        self.x = torch.randn(n, 3, *image_size)
        self.y = torch.randint(0, 2, (n,)).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class _DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=pad, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.sep1 = _DepthwiseSeparableConv2d(in_channels, out_channels)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.sep2 = _DepthwiseSeparableConv2d(out_channels, out_channels)
        self.bn2  = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        x = self.relu(x)
        x = self.bn1(self.sep1(x))
        x = self.relu(x)
        x = self.bn2(self.sep2(x))
        x = self.pool(x)
        return x + skip


class MiniXception(nn.Module):
    """PyTorch port of the mini-Xception binary classifier from the Keras Cats vs Dogs example."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        # Images are rescaled to [0, 1] by ToTensor in the dataloader transforms.
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            _ResidualBlock(128, 256),
            _ResidualBlock(256, 512),
            _ResidualBlock(512, 728),
        )
        self.top = nn.Sequential(
            _DepthwiseSeparableConv2d(728, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        units = 1 if num_classes == 2 else num_classes
        self.head = nn.Linear(1024, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.blocks(x)
        x = self.top(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_transforms(image_size: tuple, augment: bool) -> transforms.Compose:
    tfm = [transforms.Resize(image_size)]
    if augment:
        tfm += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=36),  # ≈ 0.1 × 360°
        ]
    tfm.append(transforms.ToTensor())  # scales [0, 255] → [0, 1]
    return transforms.Compose(tfm)


# ── FL Interface ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 128))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory", True)

    image_size = tuple(config.get("image_size", [180, 180]))
    data_path  = config.get("data_path", "PetImages")
    seed       = config.get("seed", 42)
    val_ratio  = config.get("val_ratio", 0.2)

    has_data = os.path.isdir(data_path) and any(
        os.path.isdir(os.path.join(data_path, c))
        for c in CatsDogsDataset.CLASSES
    )

    if not has_data:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No Cats/Dogs data found at '{data_path}'. "
                "Set config['allow_synthetic_data'] = True to use synthetic data instead."
            )
        n     = config.get("synthetic_n", 400)
        n_val = max(1, int(n * val_ratio))
        full  = _SyntheticImageDataset(n=n, image_size=image_size)
        train_ds, val_ds = random_split(
            full, [n - n_val, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        ds = train_ds if split == "train" else val_ds
    else:
        # Discover all valid samples once, then apply per-split transforms via Subset.
        base     = CatsDogsDataset(root=data_path, transform=None)
        n_total  = len(base)
        n_val    = max(1, int(n_total * val_ratio))
        n_train  = n_total - n_val
        indices  = torch.randperm(
            n_total, generator=torch.Generator().manual_seed(seed)
        ).tolist()
        if split == "train":
            base.transform = _make_transforms(image_size, augment=True)
            ds = Subset(base, indices[:n_train])
        else:
            base.transform = _make_transforms(image_size, augment=False)
            ds = Subset(base, indices[n_train:])

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
        batch   = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs  = batch[0]
        targets = batch[1]
    elif isinstance(batch, dict):
        batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    targets = targets.float()
    logits  = model(inputs)                              # shape (N, 1)
    loss    = nn.BCEWithLogitsLoss()(logits.squeeze(1), targets)
    return loss
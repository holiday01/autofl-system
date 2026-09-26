"""
Auto-generated FL client module.
Original: Keras 'Image classification from scratch' — Cats vs Dogs mini-Xception
          (fchollet, 2020/04/27, last modified 2023/11/09).

Exposes:
  build_model(config)                    -> nn.Module
  build_dataloader(config, split)        -> DataLoader
  train_step(model, batch, opt, config)  -> loss tensor (with grad_fn)

CONTRACT:
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T


# ── Model (converted from Keras mini-Xception) ──────────────────────────────

class SeparableConv2d(nn.Module):
    """
    Depthwise-separable convolution — equivalent to Keras SeparableConv2D
    with default settings (depthwise bias=False, pointwise bias=True).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _ResBlock(nn.Module):
    """
    One residual block from the mini-Xception:
        relu -> SepConv -> BN -> relu -> SepConv -> BN -> MaxPool  +  skip projection
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.sep1 = SeparableConv2d(in_channels, out_channels, 3)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.sep2 = SeparableConv2d(out_channels, out_channels, 3)
        self.bn2  = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)      # "same" padding at stride-2
        self.proj = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.proj(x)
        out = torch.relu(x)
        out = self.bn1(self.sep1(out))
        out = torch.relu(out)
        out = self.bn2(self.sep2(out))
        out = self.pool(out)
        return out + shortcut


class MiniXception(nn.Module):
    """
    PyTorch port of the Keras mini-Xception used in the Cats vs Dogs example.

    Inputs : float32 tensors in [0, 1] — shape (B, C, H, W).
             T.ToTensor() in the dataset handles the [0,255]->[0,1] rescaling
             that Keras layers.Rescaling(1./255) performed inside the model.
    Outputs: raw logits — shape (B, 1) for binary, (B, num_classes) for multi-class.
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # Entry block  (Conv 128, stride 2 -> BN -> ReLU)
        self.entry_conv = nn.Conv2d(input_channels, 128, 3, stride=2, padding=1, bias=False)
        self.entry_bn   = nn.BatchNorm2d(128)

        # Residual blocks  [256, 512, 728]
        self.block1 = _ResBlock(128, 256)
        self.block2 = _ResBlock(256, 512)
        self.block3 = _ResBlock(512, 728)

        # Top separable conv  (1024)
        self.top_sep = SeparableConv2d(728, 1024, 3)
        self.top_bn  = nn.BatchNorm2d(1024)

        # Classification head
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)
        units = 1 if num_classes == 2 else num_classes
        self.fc = nn.Linear(1024, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.entry_bn(self.entry_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.relu(self.top_bn(self.top_sep(x)))
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ── Dataset ──────────────────────────────────────────────────────────────────

class _CatsDogsDataset(Dataset):
    """
    Reads a PetImages-style directory tree:
        <root>/Cat/<*.jpg>
        <root>/Dog/<*.jpg>
    Labels: Cat=0.0, Dog=1.0  (float32 for BCEWithLogitsLoss).
    Returns tensors in [0, 1] via T.ToTensor — equivalent to Keras Rescaling(1/255).
    """
    CLASS_NAMES = ("Cat", "Dog")

    def __init__(self, root: str, image_size: tuple = (180, 180)):
        self._tf = T.Compose([T.Resize(image_size), T.ToTensor()])
        self.samples: list[str] = []
        self.labels:  list[float] = []
        for idx, cls in enumerate(self.CLASS_NAMES):
            cls_dir = os.path.join(root, cls)
            if os.path.isdir(cls_dir):
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append(os.path.join(cls_dir, fname))
                        self.labels.append(float(idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self._tf(img), torch.tensor(self.labels[idx], dtype=torch.float32)


# Keras RandomFlip("horizontal") + RandomRotation(0.1):
#   factor=0.1 is a fraction of 2π  →  0.1 × 360° = 36° max rotation.
_AUGMENT_TF = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=36),
])


class _AugmentedSubset(Dataset):
    """Wraps a Subset and applies training-time augmentation to the image tensors."""
    def __init__(self, subset):
        self.subset = subset

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        return _AUGMENT_TF(x), y


class _SyntheticDataset(Dataset):
    """Purely synthetic fallback — random pixel tensors with random binary labels."""
    def __init__(self, n: int = 200, image_size: tuple = (180, 180)):
        H, W = image_size
        self.x = torch.rand(n, 3, H, W)
        self.y = torch.randint(0, 2, (n,)).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate and return MiniXception from config."""
    kwargs = config.get("model_kwargs", {})
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Expected config keys
    --------------------
    data_path            str   directory that contains 'PetImages/' (or is PetImages itself)
    local.batch_size     int   default 16
    local.num_workers    int   default 2
    local.pin_memory     bool  default True
    image_size           list  [H, W], default [180, 180]
    val_ratio            float fraction reserved for val, default 0.2
    seed                 int   random-split seed, default 1337
    allow_synthetic_data bool  must be True to fall back to random tensors
    """
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  16))
    num_workers = local.get("num_workers", config.get("num_workers",  2))
    pin_memory  = local.get("pin_memory",  True)

    image_size = tuple(config.get("image_size", [180, 180]))
    val_ratio  = config.get("val_ratio", 0.2)
    seed       = config.get("seed", 1337)

    data_path = config.get("data_path", ".")
    pets_dir  = (
        data_path
        if os.path.basename(data_path.rstrip("/\\")) == "PetImages"
        else os.path.join(data_path, "PetImages")
    )

    has_real_data = os.path.isdir(pets_dir) and any(
        os.path.isdir(os.path.join(pets_dir, c)) for c in ("Cat", "Dog")
    )

    if has_real_data:
        full_ds = _CatsDogsDataset(pets_dir, image_size=image_size)
    else:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"PetImages dataset not found at '{pets_dir}'. "
                "Set config['data_path'] to the directory that contains the 'PetImages' folder, "
                "or set config['allow_synthetic_data'] = True to use random tensors for smoke-testing."
            )
        full_ds = _SyntheticDataset(n=200, image_size=image_size)

    n_val   = max(1, int(len(full_ds) * val_ratio))
    n_train = len(full_ds) - n_val
    train_subset, val_subset = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    if split == "train":
        ds      = _AugmentedSubset(train_subset)
        shuffle = True
    else:
        ds      = val_subset
        shuffle = False

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
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.

    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT .detach() or .item() the returned loss.

    Loss function mirrors the original Keras compile():
      binary   (num_classes=2) -> BCEWithLogitsLoss   (from_logits=True equivalent)
      multiclass               -> CrossEntropyLoss
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch  = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        inputs  = batch.get("input",  batch.get("x",      batch.get("image")))
        targets = batch.get("label",  batch.get("y",      batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    logits      = model(inputs)
    num_classes = config.get("model_kwargs", {}).get("num_classes", 2)

    if num_classes == 2:
        # Binary: squeeze (B,1) -> (B,) to match float target shape
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits.squeeze(1), targets.float())
    else:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets.long())

    return loss
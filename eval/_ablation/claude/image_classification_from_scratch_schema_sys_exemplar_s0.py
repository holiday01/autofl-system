"""
Auto-generated FL client module.
Original: Keras image classification — Cats vs Dogs mini-Xception network.
TensorFlow/Keras model converted to an equivalent PyTorch nn.Module.

Exposes:
  build_model(config)                      -> nn.Module
  build_dataloader(config, split="train")  -> DataLoader
  train_step(model, batch, opt, config)    -> loss tensor (with grad_fn)

CONTRACT:
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT .detach() it).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# ── PyTorch equivalent of the Keras mini-Xception architecture ─────────────────

class SeparableConv2d(nn.Module):
    """
    Depthwise-separable convolution — equivalent to Keras SeparableConv2D.
    Applies a per-channel (depthwise) conv followed by a 1×1 (pointwise) conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=padding,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _XceptionBlock(nn.Module):
    """
    One residual block of the mini-Xception network.
    Mirrors the Keras loop body for block sizes [256, 512, 728]:
      ReLU → SepConv → BN → ReLU → SepConv → BN → MaxPool  (+  1×1 residual proj)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.sep1 = SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.sep2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2  = nn.BatchNorm2d(out_channels)
        # MaxPool2d(3, stride=2, padding=1) reproduces Keras MaxPooling2D(3, strides=2, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 1×1 conv with stride=2 projects the residual to the same spatial size and depth
        self.residual_proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        x = self.bn1(self.sep1(F.relu(x)))
        x = self.bn2(self.sep2(F.relu(x)))
        x = self.pool(x)
        return x + residual


class MiniXception(nn.Module):
    """
    PyTorch port of the Keras mini-Xception network for binary image classification.

    Input : (B, 3, H, W) float32 with pixel values in [0, 255].
    Output: (B, 1) raw logits — pair with nn.BCEWithLogitsLoss.

    The first operation rescales pixels to [0, 1], faithfully reproducing
    the Keras ``Rescaling(1.0 / 255)`` layer that opens the original model.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()

        # Entry block: Conv2d(128, 3, stride=2, same) → BN → ReLU
        self.entry_conv = nn.Conv2d(
            input_channels, 128, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.entry_bn   = nn.BatchNorm2d(128)

        # Residual blocks for sizes [256, 512, 728]
        self.block1 = _XceptionBlock(128, 256)
        self.block2 = _XceptionBlock(256, 512)
        self.block3 = _XceptionBlock(512, 728)

        # Top separable conv: SepConv(1024, 3, same) → BN → ReLU
        self.top_sep = SeparableConv2d(728, 1024, kernel_size=3, padding=1)
        self.top_bn  = nn.BatchNorm2d(1024)

        # Classification head
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        # num_classes == 2 → single logit (matches Keras: units = 1 when num_classes == 2)
        units = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(1024, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0                                              # Rescaling(1/255)
        x = F.relu(self.entry_bn(self.entry_conv(x)))             # entry block
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.top_bn(self.top_sep(x)))                  # top conv
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)                                  # (B, 1) logits


# ── Dataset ────────────────────────────────────────────────────────────────────

class CatsDogsDataset(Dataset):
    """
    Loads images from a directory with the structure::

        <root>/Cat/<image files>
        <root>/Dog/<image files>

    Returns ``(image_tensor, binary_label)`` where the image tensor has shape
    ``(3, H, W)`` with pixel values in **[0, 255]** (the model rescales internally).
    Corrupted or unreadable files are silently replaced with zero tensors.
    """

    _VALID_EXT = {".jpg", ".jpeg", ".png"}

    def __init__(self, root: str, image_size: tuple = (180, 180)):
        self.image_size = image_size
        self.samples: list = []
        if os.path.isdir(root):
            for label_idx, cls_dir in enumerate(sorted(os.listdir(root))):
                cls_path = os.path.join(root, cls_dir)
                if not os.path.isdir(cls_path):
                    continue
                for fname in sorted(os.listdir(cls_path)):
                    if os.path.splitext(fname)[1].lower() in self._VALID_EXT:
                        self.samples.append((os.path.join(cls_path, fname), label_idx))

        # ToTensor normalises to [0,1]; multiply by 255 so the model's internal
        # Rescaling(1/255) layer sees the expected [0,255] range.
        self._to_tensor = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 255.0),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label = self.samples[idx]
        try:
            img = Image.open(fpath).convert("RGB")
            x   = self._to_tensor(img)
        except Exception:
            x = torch.zeros(3, *self.image_size)
        return x, torch.tensor(label, dtype=torch.float32)


class _AugmentedSubset(Dataset):
    """
    Wraps a Subset and applies the same augmentations as the original script:
      - Random horizontal flip   (Keras ``RandomFlip("horizontal")``)
      - Random rotation ±36°     (Keras ``RandomRotation(factor=0.1)`` → ±0.1×360°)

    torchvision ≥ 0.8 applies these transforms directly to float tensors.
    """

    def __init__(self, subset):
        self.subset = subset
        self._aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=36),
        ])

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        return self._aug(x), y


# ── FL Interface ───────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 16))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    data_path  = config.get("data_path", ".")
    image_size = tuple(config.get("image_size", [180, 180]))
    val_ratio  = config.get("val_ratio", 0.2)
    seed       = config.get("seed", 1337)

    # ── Attempt to load real data ──────────────────────────────────────────────
    real_data_available = False
    if os.path.isdir(data_path):
        full_ds = CatsDogsDataset(root=data_path, image_size=image_size)
        real_data_available = len(full_ds) > 0

    if not real_data_available:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No valid image files found under '{data_path}'. "
                "Expected subdirectories 'Cat/' and 'Dog/' containing JPEG/PNG images. "
                "Set config['allow_synthetic_data']=True to use synthetic data for testing."
            )
        # Synthetic fallback: random [0,255] RGB images with binary labels
        n_synth = 200
        X = torch.randint(0, 256, (n_synth, 3, *image_size), dtype=torch.float32)
        y = torch.randint(0, 2,   (n_synth,),                dtype=torch.float32)
        full_ds = torch.utils.data.TensorDataset(X, y)

    # ── Train / val split via random_split ────────────────────────────────────
    n_total = len(full_ds)
    n_val   = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_subset, val_subset = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    ds = _AugmentedSubset(train_subset) if split == "train" else val_subset

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
    do NOT do either here, and do NOT .detach() or .item() the returned loss.

    Loss: BCEWithLogitsLoss — equivalent to Keras
    ``BinaryCrossentropy(from_logits=True)`` used in the original script.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch   = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs  = batch[0]
        targets = batch[1]
    elif isinstance(batch, dict):
        batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        inputs  = batch.get("input",  batch.get("x", batch.get("image")))
        targets = batch.get("label",  batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # Targets must be float (B,) for BCEWithLogitsLoss
    targets = targets.float()
    if targets.dim() > 1:
        targets = targets.squeeze(-1)

    logits = model(inputs)      # (B, 1)
    logits = logits.squeeze(-1) # (B,)

    loss = nn.BCEWithLogitsLoss()(logits, targets)
    return loss   # grad_fn is attached; do NOT call .backward() or .item() here
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _is_valid_image(fpath: str) -> bool:
    """Return True only for JPEG files whose header contains the JFIF marker.

    Mirrors the corruption-filter loop from the original training script.
    """
    try:
        with open(fpath, "rb") as f:
            header = f.read(10)
        return b"JFIF" in header
    except Exception:
        return False


# ─── Model (Keras mini-Xception → PyTorch) ───────────────────────────────────

class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution; matches Keras SeparableConv2D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class MiniXception(nn.Module):
    """PyTorch port of the Keras mini-Xception network (Cats vs Dogs example).

    Architecture is a faithful translation of make_model() from the original
    script:
      • Entry block  : Conv2d(128, 3×3, stride=2) + BN + ReLU
      • Residual loop: channel sizes 256 → 512 → 728
          - ReLU → SepConv + BN → ReLU → SepConv + BN → MaxPool(3×3, stride=2)
          - Skip : Conv2d(size, 1×1, stride=2) on previous activation
      • Top block    : SepConv(1024, 3×3) + BN + ReLU
      • Head         : GlobalAvgPool → Dropout(0.25) → Linear(1)

    Input : (N, 3, H, W)  — pixel values normalised to [0, 1] by transforms.ToTensor()
    Output: raw logit tensor, shape (N, 1) for binary / (N, C) for multi-class.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()

        # Entry block
        self.entry_conv = nn.Conv2d(
            input_channels, 128, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.entry_bn = nn.BatchNorm2d(128)

        # Residual blocks  (channel sizes mirror the Keras loop: 256, 512, 728)
        self.res_blocks = nn.ModuleList()
        in_ch = 128
        for size in [256, 512, 728]:
            self.res_blocks.append(
                nn.ModuleDict({
                    "sep1": SeparableConv2d(in_ch, size, 3, padding=1),
                    "bn1":  nn.BatchNorm2d(size),
                    "sep2": SeparableConv2d(size, size, 3, padding=1),
                    "bn2":  nn.BatchNorm2d(size),
                    # MaxPool2d(3, stride=2, padding=1) gives the same spatial
                    # output as Keras MaxPooling2D(3, strides=2, padding="same")
                    "pool": nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    # Strided 1×1 conv to project the residual branch
                    "proj": nn.Conv2d(in_ch, size, 1, stride=2, bias=False),
                })
            )
            in_ch = size

        # Top block
        self.top_sep = SeparableConv2d(728, 1024, 3, padding=1)
        self.top_bn  = nn.BatchNorm2d(1024)

        # Classification head
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(0.25)
        out_units       = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(1024, out_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry block
        x    = F.relu(self.entry_bn(self.entry_conv(x)))
        prev = x

        # Residual blocks
        for blk in self.res_blocks:
            # Project the *previous* block's activation (matches Keras logic)
            residual = blk["proj"](prev)

            x = blk["sep1"](F.relu(x))
            x = blk["bn1"](x)
            x = blk["sep2"](F.relu(x))
            x = blk["bn2"](x)
            x = blk["pool"](x)

            x    = x + residual
            prev = x

        # Top block
        x = F.relu(self.top_bn(self.top_sep(x)))

        # Head: GAP → flatten → dropout → linear
        x = torch.flatten(self.gap(x), 1)
        x = self.classifier(self.dropout(x))
        return x   # raw logits — no sigmoid/softmax here


# ─── FL API ──────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the MiniXception model.

    Any keyword arguments understood by MiniXception.__init__ (e.g.
    ``input_channels``, ``num_classes``) may be passed via
    ``config["model_kwargs"]``.
    """
    kwargs = config.get("model_kwargs", {})
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the Cats vs Dogs PetImages dataset.

    Expected on-disk layout::

        <data_path>/
            PetImages/
                Cat/  <*.jpg>
                Dog/  <*.jpg>

    Corrupted images (no JFIF header) are silently excluded, matching the
    original script's filtering step.

    Raises
    ------
    FileNotFoundError
        When the PetImages directory does not exist **and**
        ``config["allow_synthetic_data"]`` is not ``True``.
    """
    local_cfg   = config.get("local", {})
    batch_size  = local_cfg.get("batch_size", 16)
    num_workers = local_cfg.get("num_workers", 2)
    data_path   = config.get("data_path", ".")
    image_size  = tuple(config.get("image_size", [180, 180]))
    val_frac    = config.get("val_fraction", 0.2)
    seed        = config.get("seed", 1337)

    pet_dir = os.path.join(data_path, "PetImages")

    # ── Real data path ────────────────────────────────────────────────────
    if os.path.isdir(pet_dir):
        # Training transforms: augmentation matching the original script
        #   RandomFlip("horizontal")  → RandomHorizontalFlip()
        #   RandomRotation(0.1)       → RandomRotation(36°)  [0.1 × 360°]
        # Validation: resize + normalise only.
        train_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=36),
            transforms.ToTensor(),          # uint8 [0,255] → float32 [0,1], HWC→CHW
        ])
        val_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        # Load without a transform first to compute stable split indices.
        index_ds = torchvision.datasets.ImageFolder(
            root=pet_dir, is_valid_file=_is_valid_image
        )
        n_total = len(index_ds)
        n_val   = max(1, int(n_total * val_frac))
        n_train = n_total - n_val
        gen = torch.Generator().manual_seed(seed)
        train_sub, val_sub = random_split(
            index_ds, [n_train, n_val], generator=gen
        )
        train_idx: list = train_sub.indices
        val_idx:   list = val_sub.indices

        # Reload with the split-appropriate transform, then slice to the indices.
        chosen_tf      = val_tf     if split == "val" else train_tf
        chosen_indices = val_idx    if split == "val" else train_idx
        ds = torchvision.datasets.ImageFolder(
            root=pet_dir, transform=chosen_tf, is_valid_file=_is_valid_image
        )
        subset = torch.utils.data.Subset(ds, chosen_indices)

        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    # ── Synthetic fallback ────────────────────────────────────────────────
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"Cats vs Dogs 'PetImages' directory not found at '{pet_dir}'. "
            "Download kagglecatsanddogs_5340.zip from "
            "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-"
            "8368-6DEBA77B919F/kagglecatsanddogs_5340.zip, unzip it, and set "
            "config['data_path'] to the extraction root — OR set "
            "config['allow_synthetic_data'] = True to run on random synthetic "
            "data for unit-testing purposes only."
        )

    # Synthetic tensors: random RGB images in [0, 1], binary labels.
    n_synth = 256
    h, w    = image_size
    images  = torch.randn(n_synth, 3, h, w)
    labels  = torch.randint(0, 2, (n_synth,))
    synth_ds = TensorDataset(images, labels)

    n_val   = max(1, int(n_synth * val_frac))
    n_train = n_synth - n_val
    gen = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(synth_ds, [n_train, n_val], generator=gen)

    chosen = val_sub if split == "val" else train_sub
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,      # accepted but intentionally unused — FL runtime owns the step
    config: dict,
) -> torch.Tensor:
    """Execute ONE forward pass and return the scalar loss (gradients attached).

    The FL runtime is responsible for calling ``loss.backward()`` and
    ``optimizer.step()``; this function must not do either.
    """
    device = next(model.parameters()).device

    images, labels = batch
    images = images.to(device)
    labels = labels.float().to(device)

    # Forward pass: logits shape (N, 1) → squeeze to (N,) for BCEWithLogitsLoss
    logits = model(images).squeeze(1)
    loss   = F.binary_cross_entropy_with_logits(logits, labels)
    return loss   # grad is attached; caller invokes loss.backward()
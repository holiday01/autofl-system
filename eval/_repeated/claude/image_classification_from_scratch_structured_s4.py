import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------------------------
# Depthwise-separable convolution  (mirrors Keras SeparableConv2D)
# ---------------------------------------------------------------------------

class _SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, padding: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


# ---------------------------------------------------------------------------
# Model — PyTorch port of the Keras mini-Xception
# (fchollet, "Image classification from scratch", 2020/last mod 2023)
# ---------------------------------------------------------------------------

class MiniXception(nn.Module):
    """
    Faithful PyTorch re-implementation of the Keras mini-Xception.

    Input: float32 tensors in [0, 1] produced by transforms.ToTensor().
    The original Rescaling(1/255) layer is replaced by that transform so
    the model itself operates on [0, 1] inputs directly.

    Architecture (preserved exactly):
        Entry  : Conv2D(128, 3, stride=2, same) → BN → ReLU
        Middle : three residual blocks with SeparableConv2D,
                 sizes = (256, 512, 728)
        Top    : SeparableConv2D(1024) → BN → ReLU →
                 GlobalAvgPool → Dropout(0.25) → Dense(1 | num_classes)
    """

    def __init__(self, input_shape: tuple = (3, 180, 180), num_classes: int = 2):
        super().__init__()
        in_channels = input_shape[0]

        # ── Entry block ────────────────────────────────────────────────────
        self.entry_conv = nn.Conv2d(in_channels, 128, 3, stride=2,
                                    padding=1, bias=False)
        self.entry_bn   = nn.BatchNorm2d(128)

        # ── Residual middle blocks ──────────────────────────────────────────
        self.sep_blocks = nn.ModuleList()
        self.res_convs  = nn.ModuleList()
        prev_ch = 128
        for size in (256, 512, 728):
            # Main branch: ReLU→SepConv→BN→ReLU→SepConv→BN→MaxPool
            self.sep_blocks.append(nn.Sequential(
                nn.ReLU(),
                _SeparableConv2d(prev_ch, size, 3, padding=1),
                nn.BatchNorm2d(size),
                nn.ReLU(),
                _SeparableConv2d(size, size, 3, padding=1),
                nn.BatchNorm2d(size),
                nn.MaxPool2d(3, stride=2, padding=1),   # 'same' pooling
            ))
            # Skip / residual branch: Conv2D(size, 1, stride=2)
            self.res_convs.append(
                nn.Conv2d(prev_ch, size, 1, stride=2, padding=0, bias=False)
            )
            prev_ch = size

        # ── Top block ──────────────────────────────────────────────────────
        self.top_sep    = _SeparableConv2d(728, 1024, 3, padding=1)
        self.top_bn     = nn.BatchNorm2d(1024)
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(0.25)
        units = 1 if num_classes == 2 else num_classes
        self.fc         = nn.Linear(1024, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry block
        x = F.relu(self.entry_bn(self.entry_conv(x)))
        prev = x

        # Residual middle blocks
        for block, res_conv in zip(self.sep_blocks, self.res_convs):
            x = block(x) + res_conv(prev)
            prev = x

        # Top block
        x = F.relu(self.top_bn(self.top_sep(x)))
        x = self.global_avg(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Dataset helper: applies per-split transforms over an ImageFolder subset
# ---------------------------------------------------------------------------

class _TransformSubset(Dataset):
    """Wraps an ImageFolder, selects a list of indices, and applies a transform."""

    def __init__(self, folder: datasets.ImageFolder,
                 indices: list, tfm: transforms.Compose):
        self.folder  = folder
        self.indices = indices
        self.tfm     = tfm

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        path, label = self.folder.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), label


# ---------------------------------------------------------------------------
# FL client API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the MiniXception model.

    Relevant config keys:
        model_kwargs : dict passed directly to MiniXception.__init__
                       Supported keys: input_shape (tuple), num_classes (int)
    """
    kwargs = config.get("model_kwargs", {})
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    config keys consumed:
        data_path            – ImageFolder-compatible root dir (default ".")
                               Must contain class sub-dirs, e.g. Cat/ and Dog/.
        local.batch_size     – mini-batch size (default 16)
        local.num_workers    – DataLoader worker processes (default 2)
        image_size           – (H, W) tuple (default (180, 180))
        val_fraction         – fraction held out for validation (default 0.2)
        seed                 – RNG seed for the deterministic split (default 1337)
        allow_synthetic_data – if True, fall back to random tensors when the
                               real dataset is unavailable (default False).
                               When False and data is missing, raises FileNotFoundError.
    """
    local_cfg   = config.get("local", {})
    batch_size  = local_cfg.get("batch_size", 16)
    num_workers = local_cfg.get("num_workers", 2)
    data_path   = config.get("data_path", ".")
    image_size  = config.get("image_size", (180, 180))
    val_frac    = config.get("val_fraction", 0.2)
    seed        = config.get("seed", 1337)

    # ── Transforms ─────────────────────────────────────────────────────────
    # Training: RandomHorizontalFlip + RandomRotation(36°)
    #   mirrors Keras: layers.RandomFlip("horizontal") + layers.RandomRotation(0.1)
    #   (0.1 fraction-of-2π ≈ 36 degrees)
    train_tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=36),
        transforms.ToTensor(),          # → float32 [0, 1]
    ])
    val_tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # ── Try to load real data ───────────────────────────────────────────────
    try:
        # Load without a transform (we apply per-split transforms in _TransformSubset)
        base_ds = datasets.ImageFolder(data_path, transform=None)
        if len(base_ds) == 0:
            raise RuntimeError("ImageFolder found 0 samples.")
    except (FileNotFoundError, RuntimeError) as exc:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Cats vs Dogs dataset not found or empty at '{data_path}'. "
                "Provide a valid ImageFolder-compatible directory (sub-dirs per class, "
                "e.g. Cat/ and Dog/), or set config['allow_synthetic_data'] = True "
                "to use random synthetic tensors for dry-run / CI testing."
            ) from exc

        # ── Synthetic fallback (dry-run / CI only) ────────────────────────
        n   = 200
        h, w = image_size
        xs  = torch.randint(0, 256, (n, 3, h, w), dtype=torch.float32) / 255.0
        ys  = torch.randint(0, 2, (n,), dtype=torch.float32)
        syn = TensorDataset(xs, ys)
        n_val   = max(1, int(n * val_frac))
        n_train = n - n_val
        gen = torch.Generator().manual_seed(seed)
        train_sub, val_sub = random_split(syn, [n_train, n_val], generator=gen)
        ds = train_sub if split == "train" else val_sub
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=(split == "train"),
                          drop_last=(split == "train"),
                          num_workers=0)

    # ── Split real dataset deterministically ────────────────────────────────
    n_total = len(base_ds)
    n_val   = max(1, int(n_total * val_frac))
    n_train = n_total - n_val

    gen  = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen).tolist()
    train_indices = perm[:n_train]
    val_indices   = perm[n_train:]

    indices = train_indices if split == "train" else val_indices
    tfm     = train_tfm    if split == "train" else val_tfm
    ds      = _TransformSubset(base_ds, indices, tfm)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Execute ONE forward pass and return the scalar loss tensor with grad attached.

    The FL runtime owns the backward pass and the optimizer step;
    this function must NOT (and does not) call loss.backward() or
    optimizer.step().

    Loss: BinaryCrossEntropyWithLogits
          (mirrors Keras: BinaryCrossentropy(from_logits=True))
    """
    device = next(model.parameters()).device

    images, labels = batch
    images = images.to(device)                          # float32 [0, 1]
    labels = labels.to(device, dtype=torch.float32)     # float32 required by BCEWithLogits

    logits = model(images)          # shape: [B, 1]  (binary head)
    logits = logits.squeeze(1)      # shape: [B]

    # Grad is attached; backward / optimizer.step handled by the FL runtime.
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Architecture — PyTorch port of the mini-Xception defined in make_model()
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv2d(nn.Module):
    """
    Equivalent to Keras SeparableConv2D:
    depthwise (groups=in_ch) → pointwise (1×1).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, padding: int = 0, stride: int = 1,
                 bias: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _XceptionResBlock(nn.Module):
    """
    One residual block from make_model():
        relu → SepConv → BN → relu → SepConv → BN → MaxPool
        + Conv1×1 projection on the skip path (applied BEFORE relu, matching Keras).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.sep1 = DepthwiseSeparableConv2d(in_channels, out_channels, 3, padding=1)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.sep2 = DepthwiseSeparableConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2  = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        # Projection applied to x before relu — mirrors Keras previous_block_activation
        self.proj = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)        # project BEFORE relu

        out = F.relu(x)
        out = self.bn1(self.sep1(out))
        out = F.relu(out)
        out = self.bn2(self.sep2(out))
        out = self.pool(out)

        return out + residual


class MiniXception(nn.Module):
    """
    Faithful PyTorch translation of make_model() from the original Keras script.

    Input:  float tensor in **[0, 255]** range, shape (B, C, H, W).
    Output: raw logits — shape (B, 1) for binary, (B, num_classes) otherwise.
    The Rescaling(1/255) from Keras is preserved as the first operation in forward().
    """

    def __init__(self, input_channels: int = 3,
                 num_classes: int = 2,
                 image_size: int = 180):
        super().__init__()

        # Entry block
        self.entry_conv = nn.Conv2d(input_channels, 128, 3, stride=2, padding=1, bias=True)
        self.entry_bn   = nn.BatchNorm2d(128)

        # Three residual blocks — sizes [256, 512, 728]
        channel_pairs = [(128, 256), (256, 512), (512, 728)]
        self.res_blocks = nn.ModuleList([
            _XceptionResBlock(ic, oc) for ic, oc in channel_pairs
        ])

        # Final SeparableConv + BN
        self.final_sep = DepthwiseSeparableConv2d(728, 1024, 3, padding=1)
        self.final_bn  = nn.BatchNorm2d(1024)

        # Classifier head
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)
        units = 1 if num_classes == 2 else num_classes
        self.fc = nn.Linear(1024, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0                      # Rescaling(1/255)

        # Entry block
        x = self.entry_conv(x)
        x = self.entry_bn(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Final conv
        x = self.final_sep(x)
        x = self.final_bn(x)
        x = F.relu(x)

        # Head
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)                  # raw logits, no activation


# ---------------------------------------------------------------------------
# Helper: wrap a Subset with its own transform
# ---------------------------------------------------------------------------

class _SubsetWithTransform(Dataset):
    """Allows train and val subsets from random_split to carry different transforms."""

    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """Instantiate MiniXception from config."""
    kwargs = dict(config.get("model_kwargs", {}))
    kwargs.setdefault("num_classes",    2)
    kwargs.setdefault("input_channels", 3)
    kwargs.setdefault("image_size",     180)
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for 'train' or 'val'.

    Looks for a PetImages/ folder (ImageFolder-compatible layout) under data_path.
    If absent and allow_synthetic_data is False  → raises FileNotFoundError.
    If absent and allow_synthetic_data is True   → uses synthetic tensors.
    """
    local_cfg   = config.get("local", {})
    batch_size  = local_cfg.get("batch_size",  16)
    num_workers = local_cfg.get("num_workers",  2)
    data_path   = config.get("data_path", ".")
    image_size  = config.get("model_kwargs", {}).get("image_size", 180)
    val_frac    = config.get("val_fraction", 0.2)

    # ── transforms ──────────────────────────────────────────────────────────
    # Pixels are kept in [0, 255] float range so that the model's internal
    # Rescaling(1/255) works identically to the original Keras version.
    _to_255 = transforms.Lambda(lambda t: t * 255.0)

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=18),   # 0.1 × 180° ≈ 18°
        transforms.ToTensor(),                   # → [0, 1]
        _to_255,                                 # → [0, 255]
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        _to_255,
    ])

    # ── real dataset ─────────────────────────────────────────────────────────
    pets_dir = os.path.join(data_path, "PetImages")
    if os.path.isdir(pets_dir):
        # Load without transform so we can apply different ones per split
        base_ds = torchvision.datasets.ImageFolder(pets_dir, transform=None)
        n_total = len(base_ds)
        n_val   = max(1, int(n_total * val_frac))
        n_train = n_total - n_val
        train_sub, val_sub = random_split(
            base_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(1337),
        )
        if split == "train":
            dataset = _SubsetWithTransform(train_sub, train_tf)
            shuffle = True
        else:
            dataset = _SubsetWithTransform(val_sub, val_tf)
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    # ── synthetic fallback — ONLY when explicitly opted in ───────────────────
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"PetImages directory not found at '{pets_dir}'. "
            "Provide the correct 'data_path' in config, or set "
            "config['allow_synthetic_data'] = True to use random tensors for "
            "dry-run / CI testing (never for real federated training)."
        )

    n_synth = local_cfg.get("synthetic_size", 256)
    # Synthetic images already in [0, 255] float range (no further transform needed)
    imgs   = torch.randint(0, 256, (n_synth, 3, image_size, image_size),
                           dtype=torch.float32)
    labels = torch.randint(0, 2, (n_synth,), dtype=torch.long)
    synth_ds = torch.utils.data.TensorDataset(imgs, labels)
    n_val    = max(1, int(n_synth * val_frac))
    n_train  = n_synth - n_val
    train_sub, val_sub = random_split(
        synth_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(1337),
    )
    chosen = train_sub if split == "train" else val_sub
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
    )


def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Single forward pass.  Returns the loss tensor with grad attached.
    The FL runtime is responsible for loss.backward() and optimizer.step().
    """
    device = next(model.parameters()).device

    imgs, labels = batch
    imgs   = imgs.to(device)
    labels = labels.to(device).float()   # BCE needs float targets

    logits = model(imgs)                 # (B, 1)
    logits = logits.squeeze(1)           # (B,)  — matches label shape

    # BinaryCrossentropy(from_logits=True) from the original Keras compile()
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss                          # grad attached; no backward/step called here
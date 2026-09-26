import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


# ---------------------------------------------------------------------------
# Helpers: separable convolution + residual block (mirror Keras originals)
# ---------------------------------------------------------------------------

class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution – equivalent to Keras SeparableConv2D."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    """One residual block from the original Keras mini-Xception architecture."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.sep1 = SeparableConv2d(in_channels,  out_channels, 3, padding=1)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.sep2 = SeparableConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2  = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        # Project residual: Conv2D(size, 1, strides=2, padding="same")
        self.proj = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.bn1(self.sep1(F.relu(x)))
        x = self.bn2(self.sep2(F.relu(x)))
        x = self.pool(x)
        return x + residual


# ---------------------------------------------------------------------------
# Model: PyTorch port of make_model() from the original Keras script
# ---------------------------------------------------------------------------

class MiniXception(nn.Module):
    """
    PyTorch equivalent of the Keras mini-Xception model.

    Input : float32 tensor (B, 3, H, W) with pixel values in [0, 255].
            Rescaling to [0, 1] is performed inside forward() to faithfully
            replicate the Keras layers.Rescaling(1./255) layer.
    Output: (B, 1) raw logits for binary classification  (num_classes == 2)
            (B, num_classes) for multi-class.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        units = 1 if num_classes == 2 else num_classes

        # Entry block
        self.entry_conv = nn.Conv2d(3, 128, 3, stride=2, padding=1, bias=False)
        self.entry_bn   = nn.BatchNorm2d(128)

        # Three residual blocks mirroring sizes=[256, 512, 728]
        self.block1 = ResidualBlock(128, 256)
        self.block2 = ResidualBlock(256, 512)
        self.block3 = ResidualBlock(512, 728)

        # Top block
        self.top_sep = SeparableConv2d(728, 1024, 3, padding=1)
        self.top_bn  = nn.BatchNorm2d(1024)

        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(0.25)
        self.classifier = nn.Linear(1024, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0                                          # Rescaling(1/255)
        x = F.relu(self.entry_bn(self.entry_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.top_bn(self.top_sep(x)))
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Internal helper: apply tensor-safe augmentation to a train Subset
# ---------------------------------------------------------------------------

class _AugmentedSubset(Dataset):
    """Wraps an existing Subset and applies an additional transform per sample."""

    def __init__(self, subset, aug_transform):
        self.subset = subset
        self.aug    = aug_transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.aug(img), label


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the MiniXception model."""
    kwargs = config.get("model_kwargs", {})
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ('train' or 'val').

    Expects the PetImages directory (Cat/ and Dog/ subdirs) to live at
    os.path.join(config['data_path'], 'PetImages').

    If the directory is absent and config['allow_synthetic_data'] is True,
    a small synthetic TensorDataset is used instead.  When the flag is False
    (the default) a FileNotFoundError is raised so that training on phantom
    data never happens silently.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path  = config.get("data_path", ".")
    image_size = tuple(config.get("image_size", (180, 180)))
    pet_path   = os.path.join(data_path, "PetImages")

    if os.path.isdir(pet_path):
        # Base transform shared by both splits – keeps pixel values in [0, 255]
        # so that MiniXception.forward() can apply the Rescaling layer faithfully.
        base_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 255.0),
        ])
        # Tensor-compatible augmentation applied only to the training split.
        # RandomFlip  → Keras layers.RandomFlip("horizontal")
        # RandomRotation(36°) → Keras layers.RandomRotation(0.1)  [0.1 × 360° = 36°]
        aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=36),
        ])

        full_ds = ImageFolder(pet_path, transform=base_tf)
        n_total = len(full_ds)
        n_val   = max(1, int(0.2 * n_total))
        n_train = n_total - n_val

        train_subset, val_subset = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(1337),
        )

        if split == "val":
            dataset = val_subset
        else:
            dataset = _AugmentedSubset(train_subset, aug_tf)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=True,
        )

    # Real dataset not found -----------------------------------------------
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"PetImages dataset not found at '{pet_path}'. "
            "Download the Kaggle Cats vs Dogs dataset and place it there, "
            "or set config['allow_synthetic_data'] = True to use synthetic "
            "data for debugging purposes only."
        )

    # Synthetic fallback (gated on allow_synthetic_data=True) ---------------
    n_samples = 200
    imgs   = torch.randint(
        0, 256, (n_samples, 3, image_size[0], image_size[1]),
        dtype=torch.float32,
    )
    labels = torch.randint(0, 2, (n_samples,), dtype=torch.float32)
    syn_ds = TensorDataset(imgs, labels)

    n_val_s   = max(1, int(0.2 * n_samples))
    n_train_s = n_samples - n_val_s
    train_s, val_s = random_split(
        syn_ds, [n_train_s, n_val_s],
        generator=torch.Generator().manual_seed(1337),
    )
    chosen = train_s if split == "train" else val_s
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(
    model: nn.Module,
    batch,
    optimizer,          # noqa: ARG001  (the FL runtime calls backward/step)
    config: dict,       # noqa: ARG001
) -> torch.Tensor:
    """
    Single forward pass for one mini-batch.

    Returns the live loss tensor (grad_fn intact).
    Does NOT call loss.backward() or optimizer.step() — the FL runtime
    handles both.
    """
    device = next(model.parameters()).device
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device).float()

    # model outputs (B, 1) logits; squeeze to (B,) for BCEWithLogitsLoss
    logits = model(images).squeeze(1)
    loss   = F.binary_cross_entropy_with_logits(logits, labels)
    return loss  # grad_fn is intact; do NOT detach or call .item()
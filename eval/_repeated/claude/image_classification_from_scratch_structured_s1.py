import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Model – PyTorch equivalent of the Keras mini-Xception architecture
# ---------------------------------------------------------------------------

class SeparableConv2d(nn.Module):
    """Depthwise-separable 2-D convolution – equivalent to Keras SeparableConv2D."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int = 0, bias: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class MiniXception(nn.Module):
    """
    PyTorch port of the mini-Xception network from
    'Image classification from scratch' (Chollet 2020).

    The original Keras model begins with a Rescaling(1/255) layer.
    That rescaling is absorbed into the dataloader transforms:
    torchvision.transforms.ToTensor already maps pixels to [0, 1],
    which is numerically identical to Rescaling(1/255).
    All other layers are reproduced exactly.

    Args:
        input_channels: number of image channels (default 3 for RGB).
        num_classes: 2 -> single logit unit (binary); >2 -> num_classes logit units.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()

        # Entry block: Conv -> BN -> ReLU
        self.entry_conv = nn.Conv2d(input_channels, 128, 3,
                                    stride=2, padding=1, bias=False)
        self.entry_bn   = nn.BatchNorm2d(128)

        # Three residual blocks with channel widths [256, 512, 728]
        self.res_blocks    = nn.ModuleList()
        self.res_shortcuts = nn.ModuleList()
        in_ch = 128
        for size in (256, 512, 728):
            self.res_blocks.append(nn.Sequential(
                nn.ReLU(),
                SeparableConv2d(in_ch,  size, 3, padding=1, bias=False),
                nn.BatchNorm2d(size),
                nn.ReLU(),
                SeparableConv2d(size, size, 3, padding=1, bias=False),
                nn.BatchNorm2d(size),
                nn.MaxPool2d(3, stride=2, padding=1),
            ))
            self.res_shortcuts.append(
                nn.Conv2d(in_ch, size, 1, stride=2, padding=0, bias=False)
            )
            in_ch = size

        # Top block: SepConv -> BN -> ReLU -> GAP -> Dropout -> FC
        self.top_conv = SeparableConv2d(728, 1024, 3, padding=1, bias=False)
        self.top_bn   = nn.BatchNorm2d(1024)
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.dropout  = nn.Dropout(0.25)
        units = 1 if num_classes == 2 else num_classes
        self.fc = nn.Linear(1024, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry block
        x    = F.relu(self.entry_bn(self.entry_conv(x)))
        prev = x

        # Residual blocks
        for block, shortcut in zip(self.res_blocks, self.res_shortcuts):
            x    = block(x) + shortcut(prev)
            prev = x

        # Top block – returns raw logits
        x = F.relu(self.top_bn(self.top_conv(x)))
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)   # (B, 1) binary or (B, C) multi-class


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the MiniXception model.

    Relevant config keys:
        model_kwargs (dict): forwarded as **kwargs to MiniXception.__init__
                             e.g. {"num_classes": 2, "input_channels": 3}
    """
    kwargs = config.get("model_kwargs", {})
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build and return a DataLoader for the Cats-vs-Dogs dataset.

    Expected on-disk layout:
        <data_path>/
            Cat/   *.jpg
            Dog/   *.jpg

    Train split: resize -> random horizontal flip -> random rotation (36 deg) -> ToTensor
    Val  split:  resize -> ToTensor

    Relevant config keys:
        local.batch_size     (int)  : mini-batch size              (default 16)
        data_path            (str)  : root folder with class dirs  (default ".")
        val_frac             (float): fraction held out for val    (default 0.2)
        image_size           (list) : [H, W] to resize images to  (default [180, 180])
        num_workers          (int)  : DataLoader worker threads    (default 2)
        allow_synthetic_data (bool) : permit synthetic fallback    (default False)

    Raises:
        FileNotFoundError: if real data is unavailable and allow_synthetic_data is False.
    """
    batch_size  = config.get("local", {}).get("batch_size", 16)
    data_path   = config.get("data_path", ".")
    val_frac    = float(config.get("val_frac", 0.2))
    image_size  = tuple(config.get("image_size", [180, 180]))
    num_workers = int(config.get("num_workers", 2))

    # Transforms mirror the original script.
    # RandomRotation factor of 0.1 in Keras means ±0.1 * 360 = ±36 degrees.
    # ToTensor() maps uint8 [0, 255] -> float32 [0, 1], replacing Rescaling(1/255).
    train_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=36),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    chosen_tf = train_tf if split == "train" else val_tf

    try:
        full_dataset = datasets.ImageFolder(data_path, transform=chosen_tf)
        if len(full_dataset) == 0:
            raise FileNotFoundError(f"No images found under {data_path!r}.")

        n_total = len(full_dataset)
        n_val   = max(1, int(n_total * val_frac))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(1337)
        train_subset, val_subset = random_split(
            full_dataset, [n_train, n_val], generator=generator
        )
        subset = train_subset if split == "train" else val_subset

    except (FileNotFoundError, RuntimeError) as exc:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset not found at {data_path!r} and "
                f"config['allow_synthetic_data'] is False.  "
                f"Provide a valid 'data_path' whose subdirectories are class labels "
                f"(e.g. Cat/, Dog/), or set config['allow_synthetic_data'] = True "
                f"to permit synthetic data for smoke-testing."
            ) from exc

        # Synthetic fallback: Gaussian noise images, random binary labels
        h, w     = image_size
        n_synth  = 200
        images   = torch.randn(n_synth, 3, h, w)
        labels   = torch.randint(0, 2, (n_synth,))
        full_dataset = TensorDataset(images, labels)
        n_val    = max(1, int(n_synth * val_frac))
        n_train  = n_synth - n_val
        generator = torch.Generator().manual_seed(1337)
        train_subset, val_subset = random_split(
            full_dataset, [n_train, n_val], generator=generator
        )
        subset = train_subset if split == "train" else val_subset

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=(split == "train"),
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
    Perform ONE forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for loss.backward() and optimizer.step();
    neither is called here.

    Loss mirrors the original Keras compile() call:
        binary (num_classes == 2): BCEWithLogitsLoss  (from_logits=True equivalent)
        multi-class               : CrossEntropyLoss
    """
    images, labels = batch
    device = next(model.parameters()).device
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    logits      = model(images)
    num_classes = config.get("model_kwargs", {}).get("num_classes", 2)

    if num_classes == 2:
        # logits shape (B, 1) -> squeeze to (B,) for BCEWithLogitsLoss
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(1), labels.float()
        )
    else:
        loss = F.cross_entropy(logits, labels.long())

    return loss  # grad attached; FL runtime calls .backward() and optimizer.step()
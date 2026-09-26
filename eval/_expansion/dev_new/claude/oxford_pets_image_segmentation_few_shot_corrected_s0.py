"""
Auto-generated FL client module.
Original script: Oxford Pets image segmentation (U-Net Xception-style, Keras → PyTorch)

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
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class OxfordPetsDataset(Dataset):
    """Oxford-IIIT Pet segmentation dataset. Falls back to synthetic data when paths are absent."""

    def __init__(self, input_img_paths, target_img_paths, img_size=(160, 160)):
        self.img_size = img_size
        self._synthetic = (not input_img_paths) or (input_img_paths[0] == "__synthetic__")
        if self._synthetic:
            self._len = len(input_img_paths) if input_img_paths else 200
        else:
            self.input_img_paths = input_img_paths
            self.target_img_paths = target_img_paths

    def __len__(self):
        return self._len if self._synthetic else len(self.input_img_paths)

    def __getitem__(self, idx):
        h, w = self.img_size
        if self._synthetic:
            return torch.randn(3, h, w), torch.randint(0, 3, (h, w))

        img = Image.open(self.input_img_paths[idx]).convert("RGB")
        img = img.resize((w, h))
        img = torch.tensor(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        mask = Image.open(self.target_img_paths[idx])
        mask = mask.resize((w, h), Image.NEAREST)
        # Ground-truth labels are 1, 2, 3 — shift to 0, 1, 2
        mask = torch.tensor(np.array(mask, dtype=np.int64) - 1)
        return img, mask


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sep1 = SeparableConv2d(in_channels, out_channels)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.sep2 = SeparableConv2d(out_channels, out_channels)
        self.bn2  = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)

    def forward(self, x):
        residual = self.residual_proj(x)
        x = self.bn1(self.sep1(F.relu(x)))
        x = self.bn2(self.sep2(F.relu(x)))
        x = self.pool(x)
        return x + residual


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.up    = nn.Upsample(scale_factor=2, mode="nearest")
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        residual = self.residual_conv(self.up(x))
        x = self.bn1(self.conv1(F.relu(x)))
        x = self.bn2(self.conv2(F.relu(x)))
        x = self.up(x)
        return x + residual


class UNetXception(nn.Module):
    def __init__(self, img_size=(160, 160), num_classes=3, in_channels=3):
        super().__init__()
        self.entry_conv = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.entry_bn   = nn.BatchNorm2d(32)

        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)

        self.up1 = UpBlock(256, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)

        # No softmax here: CrossEntropyLoss applies log_softmax internally
        self.out_conv = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.entry_bn(self.entry_conv(x)))
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out_conv(x)  # (N, num_classes, H, W)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return UNetXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    data_path   = config.get("data_path", ".")
    img_size    = tuple(config.get("img_size", [160, 160]))
    val_samples = config.get("val_samples", 1000)
    seed        = config.get("seed", 1337)

    input_dir  = os.path.join(data_path, "images")
    target_dir = os.path.join(data_path, "annotations", "trimaps")

    if os.path.isdir(input_dir) and os.path.isdir(target_dir):
        input_img_paths = sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".jpg")
        )
        target_img_paths = sorted(
            os.path.join(target_dir, f)
            for f in os.listdir(target_dir)
            if f.endswith(".png") and not f.startswith(".")
        )
        random.Random(seed).shuffle(input_img_paths)
        random.Random(seed).shuffle(target_img_paths)
    else:
        input_img_paths, target_img_paths = [], []

    if len(input_img_paths) > val_samples:
        if split == "train":
            inp = input_img_paths[:-val_samples]
            tgt = target_img_paths[:-val_samples]
        else:
            inp = input_img_paths[-val_samples:]
            tgt = target_img_paths[-val_samples:]
    else:
        inp, tgt = input_img_paths, target_img_paths

    if not inp:
        inp = tgt = ["__synthetic__"] * 200

    dataset = OxfordPetsDataset(inp, tgt, img_size=img_size)
    return DataLoader(
        dataset,
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
        targets = batch.get("label", batch.get("y", batch.get("mask")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)           # (N, num_classes, H, W)
    loss = nn.CrossEntropyLoss()(outputs, targets.long())
    return loss
"""
Auto-generated FL client module.
Original script: MONAI 3D segmentation training (UNet + DiceLoss on NIfTI volumes).

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import os
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import monai
from monai.data import list_data_collate
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)


# ── Synthetic fallback dataset ────────────────────────────────────────────────

class SyntheticVolumeDataset(Dataset):
    """Synthetic 3D volume/mask pairs for testing when real NIfTI data is unavailable."""

    def __init__(self, n: int = 40, spatial_size: tuple = (96, 96, 96)):
        self.n = n
        self.spatial_size = spatial_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.randn(1, *self.spatial_size, dtype=torch.float32)
        seg = torch.randint(0, 2, (1, *self.spatial_size), dtype=torch.float32)
        return {"img": img, "seg": seg}


# ── FL Interface ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return UNet(
        spatial_dims=kwargs.get("spatial_dims", 3),
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 1),
        channels=kwargs.get("channels", (16, 32, 64, 128, 256)),
        strides=kwargs.get("strides", (2, 2, 2, 2)),
        num_res_units=kwargs.get("num_res_units", 2),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory", True)

    data_path = config.get("data_path", ".")
    allow_synthetic = config.get("allow_synthetic_data", False)

    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs   = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if not images or not segs:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"No NIfTI image/segmentation pairs found in '{data_path}'. "
                "Set config['allow_synthetic_data'] = True to use a synthetic fallback."
            )
        spatial_size = tuple(config.get("roi_size", [96, 96, 96]))
        dataset = SyntheticVolumeDataset(
            n=config.get("synthetic_n", 40),
            spatial_size=spatial_size,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    n_total = min(len(images), len(segs))
    val_ratio = config.get("val_ratio", 0.5)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    roi_size = config.get("roi_size", [96, 96, 96])

    if split == "train":
        file_list = [
            {"img": img, "seg": seg}
            for img, seg in zip(images[:n_train], segs[:n_train])
        ]
        transforms = Compose([
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"],
                label_key="seg",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=config.get("num_samples", 4),
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
        ])
    else:
        file_list = [
            {"img": img, "seg": seg}
            for img, seg in zip(images[n_train:], segs[n_train:])
        ]
        transforms = Compose([
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"),
        ])

    dataset = monai.data.Dataset(data=file_list, transform=transforms)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list | dict,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, dict):
        inputs  = batch["img"].to(device) if isinstance(batch["img"], torch.Tensor) else batch["img"]
        targets = batch["seg"].to(device) if isinstance(batch["seg"], torch.Tensor) else batch["seg"]
    elif isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    criterion = DiceLoss(sigmoid=True)
    loss = criterion(outputs, targets)
    return loss
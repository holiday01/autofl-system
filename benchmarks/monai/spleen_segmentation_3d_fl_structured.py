"""
FL client for spleen_segmentation_3d.py (MONAI 3-D spleen segmentation).
Structured conversion: MONAI UNet with synthetic 3-D volume fallback.
"""
import os
import tempfile
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split

import monai
from monai.data import create_test_image_3d
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged,
    RandCropByPosNegLabeld, RandRotate90d, ScaleIntensityd,
)


class _SyntheticSpleenDataset(Dataset):
    """Generates synthetic 3-D image/mask pairs matching spleen_segmentation_3d specs."""

    def __init__(self, length: int = 20, spatial_size: tuple = (96, 96, 96)):
        self.length = length
        self.spatial_size = spatial_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(1, *self.spatial_size)
        seg = torch.randint(0, 2, (1, *self.spatial_size)).float()
        return {"img": img, "seg": seg}


class _NiftiSpleenDataset(Dataset):
    """Loads NIfTI image/mask pairs from a directory using MONAI transforms."""

    def __init__(self, data_dicts: list, transform):
        self._ds = monai.data.Dataset(data=data_dicts, transform=transform)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


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
    batch_size = config.get("local", {}).get("batch_size", 2)
    data_path = config.get("data_path", ".")

    imgs = sorted(
        p for p in [os.path.join(data_path, f"img{i}.nii.gz") for i in range(40)]
        if os.path.isfile(p)
    )
    segs = sorted(
        p for p in [os.path.join(data_path, f"seg{i}.nii.gz") for i in range(40)]
        if os.path.isfile(p)
    )

    if len(imgs) >= 4 and len(imgs) == len(segs):
        data_dicts = [{"img": i, "seg": s} for i, s in zip(imgs, segs)]
        train_transform = Compose([
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg",
                spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=2,
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
        ])
        val_transform = Compose([
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"),
        ])
        n_val = max(1, int(0.2 * len(data_dicts)))
        n_train = len(data_dicts) - n_val
        train_dicts, val_dicts = data_dicts[:n_train], data_dicts[n_train:]
        transform = train_transform if split == "train" else val_transform
        dicts = train_dicts if split == "train" else val_dicts
        dataset = _NiftiSpleenDataset(dicts, transform)
    else:
        logging.warning("NIfTI data not found at '%s'. Using synthetic fallback.", data_path)
        full_ds = _SyntheticSpleenDataset(length=20)
        n_val = max(1, int(0.2 * len(full_ds)))
        train_ds, val_ds = random_split(
            full_ds, [len(full_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        dataset = train_ds if split == "train" else val_ds

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=False,
    )


def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """One forward pass — returns DiceLoss with grad_fn attached.
    The FL runtime is responsible for loss.backward() and optimizer.step().
    """
    device = next(model.parameters()).device
    if isinstance(batch, dict):
        imgs = batch["img"].to(device)
        segs = batch["seg"].to(device)
    else:
        imgs, segs = batch[0].to(device), batch[1].to(device)

    outputs = model(imgs)
    loss_fn = DiceLoss(sigmoid=True)
    return loss_fn(outputs, segs)

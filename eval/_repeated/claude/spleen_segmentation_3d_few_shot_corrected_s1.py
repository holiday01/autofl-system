"""
Auto-generated FL client module.
Original script: MONAI 3D UNet segmentation training script.

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
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
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import monai
from monai.data import create_test_image_3d, list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)


def _make_synthetic_nifti_dir(tempdir: str, n: int = 40) -> tuple[list, list]:
    for i in range(n):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
        nib.save(nib.Nifti1Image(im, np.eye(4)), os.path.join(tempdir, f"img{i:d}.nii.gz"))
        nib.save(nib.Nifti1Image(seg, np.eye(4)), os.path.join(tempdir, f"seg{i:d}.nii.gz"))
    images = sorted(glob(os.path.join(tempdir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))
    return images, segs


def _build_file_dicts(data_path: str, val_ratio: float = 0.2) -> tuple[list, list]:
    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))
    if not images:
        return [], []
    n_train = len(images) - max(1, int(len(images) * val_ratio))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:n_train], segs[:n_train])]
    val_files   = [{"img": img, "seg": seg} for img, seg in zip(images[n_train:], segs[n_train:])]
    return train_files, val_files


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return monai.networks.nets.UNet(
        spatial_dims  = kwargs.get("spatial_dims", 3),
        in_channels   = kwargs.get("in_channels", 1),
        out_channels  = kwargs.get("out_channels", 1),
        channels      = tuple(kwargs.get("channels", (16, 32, 64, 128, 256))),
        strides       = tuple(kwargs.get("strides", (2, 2, 2, 2))),
        num_res_units = kwargs.get("num_res_units", 2),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory",  True)
    val_ratio   = config.get("val_ratio", 0.2)
    data_path   = config.get("data_path", "")
    spatial_size = config.get("spatial_size", [96, 96, 96])

    train_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys="img"),
        RandCropByPosNegLabeld(
            keys=["img", "seg"], label_key="seg",
            spatial_size=spatial_size, pos=1, neg=1, num_samples=4,
        ),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys="img"),
    ])

    train_files, val_files = _build_file_dicts(data_path, val_ratio=val_ratio)
    if not train_files:
        _tmpdir = tempfile.mkdtemp()
        images, segs = _make_synthetic_nifti_dir(_tmpdir, n=config.get("synthetic_n", 40))
        n_train = len(images) - max(1, int(len(images) * val_ratio))
        train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:n_train], segs[:n_train])]
        val_files   = [{"img": img, "seg": seg} for img, seg in zip(images[n_train:], segs[n_train:])]

    if split == "train":
        ds, shuffle, bs = monai.data.Dataset(data=train_files, transform=train_transforms), True, batch_size
    else:
        ds, shuffle, bs = monai.data.Dataset(data=val_files,   transform=val_transforms),  False, 1

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
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
        inputs, targets = batch[0].to(device), batch[1].to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    loss = monai.losses.DiceLoss(sigmoid=True)(outputs, targets)
    return loss
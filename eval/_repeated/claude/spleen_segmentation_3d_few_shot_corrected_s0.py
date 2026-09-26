"""
Auto-generated FL client module.
Original script: MONAI 3D segmentation training example (segmentation3d_dict).

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
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import monai
from monai.data import list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return monai.networks.nets.UNet(
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
    val_ratio = config.get("val_ratio", 0.5)

    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs   = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if images:
        n_val   = max(1, int(len(images) * val_ratio))
        n_train = len(images) - n_val
        if split == "train":
            file_list = [{"img": img, "seg": seg}
                         for img, seg in zip(images[:n_train], segs[:n_train])]
        else:
            file_list = [{"img": img, "seg": seg}
                         for img, seg in zip(images[n_train:], segs[n_train:])]
    else:
        # synthetic fallback for testing (creates temp NIfTI data in memory)
        import tempfile, nibabel as nib, numpy as np
        from monai.data import create_test_image_3d
        _tmpdir = tempfile.mkdtemp()
        n_total = config.get("synthetic_n", 10)
        for i in range(n_total):
            im, seg = create_test_image_3d(64, 64, 64, num_seg_classes=1, channel_dim=-1)
            nib.save(nib.Nifti1Image(im,  np.eye(4)), os.path.join(_tmpdir, f"img{i}.nii.gz"))
            nib.save(nib.Nifti1Image(seg, np.eye(4)), os.path.join(_tmpdir, f"seg{i}.nii.gz"))
        images = sorted(glob(os.path.join(_tmpdir, "img*.nii.gz")))
        segs   = sorted(glob(os.path.join(_tmpdir, "seg*.nii.gz")))
        n_val  = max(1, n_total // 5)
        if split == "train":
            file_list = [{"img": img, "seg": seg}
                         for img, seg in zip(images[:-n_val], segs[:-n_val])]
        else:
            file_list = [{"img": img, "seg": seg}
                         for img, seg in zip(images[-n_val:], segs[-n_val:])]

    crop_kwargs = config.get("crop_kwargs", {})
    spatial_size = crop_kwargs.get("spatial_size", [96, 96, 96])

    if split == "train":
        transforms = Compose([
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"],
                label_key="seg",
                spatial_size=spatial_size,
                pos=crop_kwargs.get("pos", 1),
                neg=crop_kwargs.get("neg", 1),
                num_samples=crop_kwargs.get("num_samples", 4),
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
        ])
    else:
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
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs  = batch.get("img", batch.get("image", batch.get("input")))
        targets = batch.get("seg", batch.get("label", batch.get("mask")))
    elif isinstance(batch, (list, tuple)):
        batch   = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    criterion = monai.losses.DiceLoss(sigmoid=True)
    loss = criterion(outputs, targets)
    return loss
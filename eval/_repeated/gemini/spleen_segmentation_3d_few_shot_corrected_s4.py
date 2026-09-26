"""
Auto-generated FL client module for MONAI 3D Segmentation.
Original script: based on a MONAI example for 3D segmentation.

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
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob
import nibabel as nib

# MONAI imports
import monai
from monai.data import create_test_image_3d, list_data_collate
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)
from monai.losses import DiceLoss
import monai.networks.nets


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Builds and returns the MONAI UNet model."""
    kwargs = config.get("model_kwargs", {})
    # Default parameters from the original script
    default_kwargs = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
    }
    final_kwargs = {**default_kwargs, **kwargs}
    return monai.networks.nets.UNet(**final_kwargs)


def _generate_synthetic_monai_data(data_dir: str, num_samples: int = 40,
                                   img_size=(128, 128, 128), num_seg_classes=1):
    """Generates synthetic 3D NIfTI image and segmentation files."""
    print(f"Generating synthetic MONAI data to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    for i in range(num_samples):
        im, seg = create_test_image_3d(img_size[0], img_size[1], img_size[2],
                                       num_seg_classes=num_seg_classes, channel_dim=-1)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(data_dir, f"img{i:03d}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(data_dir, f"seg{i:03d}.nii.gz"))
    print(f"Finished generating {num_samples} synthetic samples.")


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Builds and returns the MONAI DataLoader."""
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory = local.get("pin_memory", True)

    data_path = config.get("data_path", "./monai_data")
    num_total_samples = config.get("num_synthetic_samples", 40) # Total samples if generating

    # Check for existing data; if not found, generate synthetic data
    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if not images or not segs or len(images) != num_total_samples or len(segs) != num_total_samples:
        _generate_synthetic_monai_data(data_path, num_samples=num_total_samples)
        images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
        segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if not images or not segs:
        raise RuntimeError(f"No image/segmentation files found in {data_path} after generation attempt.")

    # Determine train/val split for the client's local dataset
    # Original script uses first 20 for train, last 20 for val (from 40 total)
    train_ratio = config.get("train_ratio", 0.5) # Ratio of total data to be used for the training split
    n_train_data_sources = int(len(images) * train_ratio)
    
    full_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
    
    # Adhere to the original script's fixed split of the *source files* for train/val
    train_source_files = full_files[:n_train_data_sources]
    val_source_files = full_files[n_train_data_sources:]

    # Define transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"),
        ]
    )

    if split == "train":
        monai_ds = monai.data.Dataset(data=train_source_files, transform=train_transforms)
    elif split == "val":
        monai_ds = monai.data.Dataset(data=val_source_files, transform=val_transforms)
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'val'.")

    return DataLoader(
        monai_ds,
        batch_size=batch_size,
        shuffle=(split == "train"), # Only shuffle training data
        num_workers=num_workers,
        collate_fn=list_data_collate, # Important for MONAI Dictionary data
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: dict | list | tuple,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass for MONAI 3D segmentation. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, labels = batch[0], batch[1] # Assuming standard (input, label) tuple
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs = batch.get("img")
        labels = batch.get("seg")
        if inputs is None or labels is None:
            raise KeyError("Expected 'img' and 'seg' keys in batch dictionary for MONAI data.")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    loss_function = DiceLoss(sigmoid=True)

    outputs = model(inputs)
    loss = loss_function(outputs, labels)
    return loss
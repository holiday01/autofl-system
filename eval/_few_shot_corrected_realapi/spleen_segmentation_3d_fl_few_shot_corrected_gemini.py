"""
Auto-generated FL client module.
Original script: monai_unet_segmentation3d.py

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
from torch.utils.data import DataLoader, random_split

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
import nibabel as nib
from glob import glob


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    # Default UNet parameters from the original script
    default_model_kwargs = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
    }
    # Merge defaults with config-provided kwargs
    model_params = {**default_model_kwargs, **kwargs}
    return monai.networks.nets.UNet(**model_params)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory", True)

    data_path = config.get("data_path", None)
    generate_synthetic_data = config.get("generate_synthetic_data", False)

    # Determine the root directory for data (either existing or for synthetic generation)
    current_data_root = data_path
    if not current_data_root:
        # If no data_path, use a local directory for synthetic data
        current_data_root = os.path.join(os.getcwd(), "synthetic_monai_data")
        os.makedirs(current_data_root, exist_ok=True)
        print(f"No data_path provided, using local directory for data: {current_data_root}")

    # Check if data already exists in current_data_root
    existing_images = sorted(glob(os.path.join(current_data_root, "img*.nii.gz")))
    existing_segs = sorted(glob(os.path.join(current_data_root, "seg*.nii.gz")))

    if generate_synthetic_data or not (existing_images and existing_segs and len(existing_images) == len(existing_segs) >= 40):
        # If explicitly asked to generate, or if data_path is empty/insufficient, generate synthetic data
        print(f"Generating 40 synthetic data samples to {current_data_root}...")
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(current_data_root, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(current_data_root, f"seg{i:d}.nii.gz"))
        print("Synthetic data generation complete.")

    # Now load the files (either newly generated or pre-existing)
    images = sorted(glob(os.path.join(current_data_root, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(current_data_root, "seg*.nii.gz")))
    all_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    if not all_files:
        raise ValueError(f"No image/segmentation pairs found at {current_data_root}. "
                         "Ensure data exists or set 'generate_synthetic_data: true' in config.")

    # Split into train/val based on the original script's logic (20 train, 20 val from 40 total)
    num_total_files = len(all_files)
    if num_total_files >= 40:
        train_files = all_files[:20]
        val_files = all_files[20:40]
    else:
        # Fallback for fewer than 40 files, or if a different split is desired
        val_ratio = config.get("val_ratio", 0.5)
        n_val = max(1, int(num_total_files * val_ratio))
        n_train = num_total_files - n_val
        train_files = all_files[:n_train]
        val_files = all_files[n_train:]

    print(f"DataLoader for split '{split}': {len(train_files if split == 'train' else val_files)} samples.")

    # Define transforms for image and segmentation
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

    ds_files = train_files if split == "train" else val_files
    ds_transforms = train_transforms if split == "train" else val_transforms

    monai_dataset = monai.data.Dataset(data=ds_files, transform=ds_transforms)

    return DataLoader(
        monai_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    inputs, labels = batch["img"].to(device), batch["seg"].to(device)

    outputs = model(inputs)

    criterion = monai.losses.DiceLoss(sigmoid=True)
    loss = criterion(outputs, labels)
    return loss
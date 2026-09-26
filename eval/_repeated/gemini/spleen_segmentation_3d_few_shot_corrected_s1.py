"""
Auto-generated FL client module.
Original script: monai/examples/segmentation3d_pytorch_lightning.py (adapted)

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
import nibabel as nib
import torch
import torch.nn as nn
from glob import glob
from torch.utils.data import DataLoader

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
from monai.networks.nets import UNet

# ── Helper for synthetic data generation ────────────────────────────────

def _generate_synthetic_data_monai(output_dir: str, num_samples: int = 40):
    """
    Generates synthetic MONAI 3D image and segmentation NIfTI files.
    This is used as a fallback if no data is found at the specified data_path.
    """
    print(f"Generating {num_samples} synthetic MONAI 3D image/segmentation pairs to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
        n_img = nib.Nifti1Image(im, np.eye(4))
        nib.save(n_img, os.path.join(output_dir, f"img{i:d}.nii.gz"))
        n_seg = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n_seg, os.path.join(output_dir, f"seg{i:d}.nii.gz"))
    print("Synthetic data generation complete.")


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the MONAI UNet model.
    Model parameters are taken from the 'model_kwargs' section in the config.
    """
    device = config.get("device", "cpu")
    model_kwargs = config.get("model_kwargs", {})
    
    # Default UNet parameters from the original script
    default_model_kwargs = dict(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    # Update with any specific kwargs from config
    unet_kwargs = {**default_model_kwargs, **model_kwargs}
    
    model = UNet(**unet_kwargs).to(device)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a PyTorch DataLoader for the specified split (train/val).
    Handles synthetic data generation if no real data is found.
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory = local.get("pin_memory", True)

    data_path = config.get("data_path", "./monai_synthetic_data")
    num_synthetic_samples = config.get("num_synthetic_samples", 40)
    
    # Check if data_path exists and contains image files. If not, generate synthetic data.
    image_files = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    if not image_files:
        _generate_synthetic_data_monai(data_path, num_samples=num_synthetic_samples)
        image_files = sorted(glob(os.path.join(data_path, "img*.nii.gz"))) # Re-glob after generation

    segs_files = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if not image_files or not segs_files or len(image_files) != len(segs_files):
        raise RuntimeError(
            f"Data inconsistency: found {len(image_files)} images and {len(segs_files)} segmentations "
            f"in {data_path}. Ensure data_path is correct or synthetic data generation worked."
        )

    all_files = [{"img": img, "seg": seg} for img, seg in zip(image_files, segs_files)]

    # Split data based on original script's logic (20 train, 20 val from 40 samples)
    # Configurable val_ratio, defaults to 0.5 to match the original example if num_synthetic_samples=40
    total_samples = len(all_files)
    val_ratio = config.get("val_ratio", 0.5)
    n_val = max(1, int(total_samples * val_ratio))
    n_train = total_samples - n_val

    if split == "train":
        data_to_load = all_files[:n_train]
        shuffle = True
        transforms = Compose(
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
    elif split == "val":
        data_to_load = all_files[-n_val:]
        shuffle = False
        transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                ScaleIntensityd(keys="img"),
            ]
        )
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'val'.")

    if not data_to_load:
        raise ValueError(f"No data files found for split '{split}'. Check data_path and config settings.")

    dataset = monai.data.Dataset(data=data_to_load, transform=transforms)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: dict, # MONAI DataLoader batches are typically dictionaries
    optimizer,  # Ignored as per contract
    config: dict,
) -> torch.Tensor:
    """
    Performs ONE forward pass and returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device # Get device from model
    
    inputs = batch["img"].to(device)
    labels = batch["seg"].to(device)

    # Loss function from the original script
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    
    outputs = model(inputs)
    loss = loss_function(outputs, labels)
    
    return loss
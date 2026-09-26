"""
Auto-generated FL client module.
Original script: MONAI 3D Segmentation (monai/examples/segmentation_3d/inference_array.py)

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

import monai
from monai.data import list_data_collate, Dataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
)


# --- Helper for data collection, assuming data_path contains Nifti files ---
def _get_data_files(root_path: str) -> list[dict]:
    """Collects image and segmentation file paths from a given root directory."""
    images = sorted(glob(os.path.join(root_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(root_path, "seg*.nii.gz")))

    if not images or not segs:
        raise FileNotFoundError(f"No Nifti image/segmentation files found in '{root_path}'. "
                                "Please ensure data_path contains 'img*.nii.gz' and 'seg*.nii.gz'.")

    # Ensure image and segmentation counts match
    if len(images) != len(segs):
        raise ValueError(f"Mismatch in image ({len(images)}) and segmentation ({len(segs)}) file counts in '{root_path}'.")

    return [{"img": img, "seg": seg} for img, seg in zip(images, segs)]


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Builds and returns the MONAI UNet model."""
    kwargs = config.get("model_kwargs", {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
    })
    return UNet(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Builds and returns a MONAI DataLoader for the specified split."""
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory = local.get("pin_memory", True)
    seed = config.get("seed", 42)
    val_ratio = config.get("val_ratio", 0.2) # Default to 20% for validation

    data_path = config.get("data_path", ".")
    all_files = _get_data_files(data_path)

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

    # Split the file list for train/validation
    num_total = len(all_files)
    indices = list(range(num_total))
    
    # Use torch.Generator for reproducible splitting
    generator = torch.Generator().manual_seed(seed)
    
    # Randomly shuffle indices and split
    shuffled_indices = [indices[i] for i in torch.randperm(num_total, generator=generator)]
    
    n_val = max(1, int(num_total * val_ratio))
    train_idx = shuffled_indices[n_val:]
    val_idx = shuffled_indices[:n_val]
    
    train_files_split = [all_files[i] for i in train_idx]
    val_files_split = [all_files[i] for i in val_idx]

    if split == "train":
        ds = Dataset(data=train_files_split, transform=train_transforms)
        shuffle_data = True
    elif split == "val":
        ds = Dataset(data=val_files_split, transform=val_transforms)
        shuffle_data = False # Validation data is typically not shuffled
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'val'.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: dict, # MONAI DataLoaders typically yield dictionaries
    optimizer, # Included for contract, but not used within this function
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device # Get model's device

    # Extract inputs and targets from the MONAI batch dictionary and move to device
    inputs = batch["img"].to(device)
    targets = batch["seg"].to(device)

    outputs = model(inputs)
    
    # Instantiate loss function with parameters from original script
    loss_function = DiceLoss(sigmoid=True)
    loss = loss_function(outputs, targets)
    
    return loss
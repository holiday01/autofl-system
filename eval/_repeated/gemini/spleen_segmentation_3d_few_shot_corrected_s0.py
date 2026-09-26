"""
Auto-generated FL client module.
Original script: monai/examples/segmentation3d_dataloader_dict.py

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
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset # Added Dataset for potential synthetic

import monai
from monai.data import list_data_collate
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    MapTransform, # For custom synthetic transform
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss


# --- Helper for synthetic data generation within the FL module ---
class LoadSyntheticImaged(MapTransform):
    """
    Custom MONAI transform to generate synthetic 3D image and segmentation data
    in-memory, mimicking the output format of LoadImaged but without file I/O.
    This is used when no real data files are found.
    """
    def __init__(self, keys, spatial_size=(128, 128, 128), num_seg_classes=1, seed=42):
        super().__init__(keys)
        self.spatial_size = spatial_size
        self.num_seg_classes = num_seg_classes
        self.rng = np.random.default_rng(seed)

    def __call__(self, data):
        # `data` here would be the item from the `data_list` (e.g., {"_idx": i})
        d = dict(data)
        
        # Generate random image data (mimic a 3D image with 1 channel, no batch dim yet)
        img = self.rng.rand(*self.spatial_size).astype(np.float32)
        
        # Create a simple synthetic segmentation mask (e.g., a sphere)
        seg = np.zeros(self.spatial_size, dtype=np.float32)
        center = np.array(self.spatial_size) / 2
        radius = self.spatial_size[0] / 4
        x, y, z = np.ogrid[:self.spatial_size[0], :self.spatial_size[1], :self.spatial_size[2]]
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        seg[distance < radius] = 1.0
        
        # Assign generated numpy arrays to the specified keys
        d["img"] = img
        d["seg"] = seg
        return d


# Define common transforms part (applies to both real and synthetic)
_common_transforms_part = [
    EnsureChannelFirstd(keys=["img", "seg"]),
    ScaleIntensityd(keys="img"),
]

# Define augmentation transforms part (for training split only)
_train_augmentation_transforms_part = [
    RandCropByPosNegLabeld(
        keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
    ),
    RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
]


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    # Default UNet parameters from the original script
    default_unet_kwargs = dict(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    # Merge defaults with any provided config
    unet_kwargs = {**default_unet_kwargs, **kwargs}
    return UNet(**unet_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local_cfg = config.get("local", {})
    batch_size = local_cfg.get("batch_size", config.get("batch_size", 2)) # Default from script
    num_workers = local_cfg.get("num_workers", config.get("num_workers", 4)) # Default from script
    pin_memory = local_cfg.get("pin_memory", True)
    seed = config.get("seed", 42)
    val_ratio = config.get("val_ratio", 0.5) # Original script uses 20 train, 20 val from 40 total

    data_path = config.get("data_path", ".")
    all_data_items = []
    is_synthetic = False

    # Try to find real Nifti files
    image_files = sorted(glob.glob(os.path.join(data_path, "img*.nii.gz")))
    seg_files = sorted(glob.glob(os.path.join(data_path, "seg*.nii.gz")))

    if image_files and seg_files and len(image_files) == len(seg_files):
        all_data_items = [{"img": img, "seg": seg} for img, seg in zip(image_files, seg_files)]
        print(f"MONAI FL Client: Found {len(all_data_items)} image/segmentation pairs in '{data_path}'")
    else:
        is_synthetic = True
        num_synthetic_samples = config.get("dataset_kwargs", {}).get("num_samples", 40)
        # Create dummy entries for monai.data.Dataset to iterate over for synthetic data
        all_data_items = [{"_idx": i} for i in range(num_synthetic_samples)] 
        print(f"MONAI FL Client: No real Nifti files found in '{data_path}' or mismatch. Using {num_synthetic_samples} synthetic samples.")

    # Split the data_list into train and validation portions (using a seeded permutation for reproducibility)
    rng_list_split = np.random.default_rng(seed)
    n_total = len(all_data_items)
    indices = rng_list_split.permutation(n_total).tolist()
    
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Select the data items for the current split
    current_data_list = [all_data_items[i] for i in train_indices] if split == "train" else [all_data_items[i] for i in val_indices]
    shuffle_data = (split == "train")

    # Construct transforms dynamically based on data source and split type
    current_transforms = []
    synthetic_kwargs = config.get("dataset_kwargs", {}) # Pass spatial_size etc. to synthetic loader

    if is_synthetic:
        # For synthetic data, use LoadSyntheticImaged.
        # Use a different seed for the validation split to ensure different synthetic data if desired.
        current_transforms.append(
            LoadSyntheticImaged(
                keys=["img", "seg"],
                spatial_size=synthetic_kwargs.get("spatial_size", (128, 128, 128)),
                num_seg_classes=synthetic_kwargs.get("num_seg_classes", 1),
                seed=seed + (0 if split == "train" else 1)
            )
        )
    else:
        # For real files, use LoadImaged.
        current_transforms.append(LoadImaged(keys=["img", "seg"]))
    
    # Add common transforms
    current_transforms.extend(_common_transforms_part)
    
    # Add augmentation transforms only for the training split
    if split == "train":
        current_transforms.extend(_train_augmentation_transforms_part)
    
    composed_transforms = Compose(current_transforms)

    # Create the MONAI Dataset
    ds = monai.data.Dataset(data=current_data_list, transform=composed_transforms)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=num_workers,
        collate_fn=list_data_collate, # MONAI specific collate_fn is important
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: dict, # MONAI DataLoaders typically yield dicts
    optimizer, # Not directly used in train_step, but kept for signature compliance
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    
    # Extract inputs and targets from the batch dictionary
    inputs = batch["img"].to(device)
    targets = batch["seg"].to(device)

    outputs = model(inputs)
    
    # Instantiate loss function (DiceLoss from original script)
    loss_kwargs = config.get("loss_kwargs", {})
    criterion = DiceLoss(sigmoid=True, **loss_kwargs)

    loss = criterion(outputs, targets)
    return loss
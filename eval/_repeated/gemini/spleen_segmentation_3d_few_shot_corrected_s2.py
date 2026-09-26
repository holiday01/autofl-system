"""
Auto-generated FL client module.
Original script: monai/examples/segmentation3d_dataloader.py

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
from torch.utils.data import Dataset, DataLoader
from glob import glob

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
from monai.losses import DiceLoss


# Custom Dataset to handle both real (file paths) and synthetic (in-memory numpy arrays) data
class MonaiLocalDataset(Dataset):
    def __init__(
        self,
        root: str,
        n_synthetic: int = 40,
        img_size=(128, 128, 128),
        num_seg_classes=1,
        channel_dim=-1,
    ):
        self.data_items = []  # Stores dicts of {key: np.ndarray or file_path}
        self.using_file_paths = False

        # Try to load real data (paths)
        if os.path.isdir(root):
            images = sorted(glob(os.path.join(root, "img*.nii.gz")))
            segs = sorted(glob(os.path.join(root, "seg*.nii.gz")))
            if images and segs and len(images) == len(segs):
                self.data_items = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
                self.using_file_paths = True

        # If no real data or root doesn't exist, generate synthetic (in-memory numpy arrays)
        if not self.data_items:
            print(f"No data found at {root}. Generating {n_synthetic} synthetic samples in memory.")
            for _ in range(n_synthetic):
                im, seg = create_test_image_3d(
                    *img_size, num_seg_classes=num_seg_classes, channel_dim=channel_dim
                )
                self.data_items.append({"img": im, "seg": seg})
            self.using_file_paths = False  # Explicitly state we are not using paths

        if not self.data_items:
            raise RuntimeError(f"Could not find or generate any data for {root}.")
        print(f"Total dataset samples: {len(self.data_items)}")

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.data_items[idx]


def _build_transforms(config: dict, split: str, using_file_paths: bool):
    """Helper to build transforms based on config and data type."""
    t_list = []
    if using_file_paths:
        t_list.append(LoadImaged(keys=["img", "seg"]))

    t_list.append(EnsureChannelFirstd(keys=["img", "seg"]))
    t_list.append(ScaleIntensityd(keys="img"))

    if split == "train":
        # RandCropByPosNegLabeld is specific to training
        t_list.append(
            RandCropByPosNegLabeld(
                keys=["img", "seg"],
                label_key="seg",
                spatial_size=[96, 96, 96],
                pos=1,
                neg=1,
                num_samples=4,
            )
        )
        t_list.append(RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]))

    return Compose(t_list)


# ── FL Interface ────────────────────────────────────────────────────────


def build_model(config: dict) -> nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    # Defaults based on original script
    kwargs = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
    }
    kwargs.update(model_kwargs)
    return UNet(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local_config = config.get("local", {})
    data_path = config.get("data_path", ".")
    batch_size = local_config.get("batch_size", config.get("batch_size", 2))
    num_workers = local_config.get("num_workers", config.get("num_workers", 4))
    pin_memory = local_config.get("pin_memory", True)
    n_synthetic_samples = config.get("n_synthetic_samples", 40)

    dataset_kwargs = config.get("dataset_kwargs", {})
    full_dataset_provider = MonaiLocalDataset(
        root=data_path,
        n_synthetic=n_synthetic_samples,
        **dataset_kwargs,
    )

    # Manual split to mimic original script's behavior (first half for train, second for val)
    total_samples = len(full_dataset_provider)
    train_end_idx = total_samples // 2  # e.g., 20 for 40 samples

    if split == "train":
        data_for_split = full_dataset_provider.data_items[:train_end_idx]
    elif split == "val":
        data_for_split = full_dataset_provider.data_items[train_end_idx:]
    else:
        raise ValueError(f"Invalid split: {split}")

    transforms = _build_transforms(config, split, full_dataset_provider.using_file_paths)

    monai_dataset = monai.data.Dataset(
        data=data_for_split,
        transform=transforms,
    )

    return DataLoader(
        monai_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),  # Shuffle only for training
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

    # MONAI's DataLoader with list_data_collate returns a dictionary where
    # each value is a batched tensor (batch_size, C, H, W, D).
    if not isinstance(batch, dict):
        raise TypeError(f"Unsupported batch type: {type(batch)}. Expected dict from MONAI DataLoader.")

    # Move batch tensors to device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    inputs = batch["img"]
    targets = batch["seg"]

    outputs = model(inputs)
    
    # Loss function from original script: monai.losses.DiceLoss(sigmoid=True)
    criterion_kwargs = config.get("loss_kwargs", {"sigmoid": True})
    criterion = DiceLoss(**criterion_kwargs)

    loss = criterion(outputs, targets)
    return loss
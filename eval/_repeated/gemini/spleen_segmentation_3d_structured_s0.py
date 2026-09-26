import logging
import os
import sys
import tempfile
from glob import glob
import math

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)
from monai.visualize import plot_2d_or_3d_image


# Define the MONAI Dataset class wrapper to include transforms
class MonaiDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=model_kwargs.get("spatial_dims", 3),
        in_channels=model_kwargs.get("in_channels", 1),
        out_channels=model_kwargs.get("out_channels", 1),
        channels=model_kwargs.get("channels", (16, 32, 64, 128, 256)),
        strides=model_kwargs.get("strides", (2, 2, 2, 2)),
        num_res_units=model_kwargs.get("num_res_units", 2),
    ).to(device)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if not images or not segs:
        if allow_synthetic_data:
            print(f"Generating synthetic data to {data_path} (this may take a while)")
            os.makedirs(data_path, exist_ok=True)
            for i in range(40):
                im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
                n = nib.Nifti1Image(im, np.eye(4))
                nib.save(n, os.path.join(data_path, f"img{i:d}.nii.gz"))
                n = nib.Nifti1Image(seg, np.eye(4))
                nib.save(n, os.path.join(data_path, f"seg{i:d}.nii.gz"))
            images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
            segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))
        else:
            raise FileNotFoundError(
                f"No NIfTI files found in {data_path}. Set 'allow_synthetic_data: true' in config to generate synthetic data."
            )

    all_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    # Use random_split to create train/val subsets
    num_total_samples = len(all_files)
    if num_total_samples < 2:
        raise ValueError("Not enough data samples for train/validation split. Need at least 2.")
    
    # Split approximately 50/50 for train/val from the whole dataset
    train_size = int(math.ceil(num_total_samples * 0.5))
    val_size = num_total_samples - train_size
    
    # Ensure sizes are positive
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Could not create train/val splits with sizes {train_size}/{val_size} from {num_total_samples} samples.")

    # Always split deterministically for reproducibility across clients if needed,
    # but for typical FL, clients might just have their own local data.
    # For now, let's just make sure we get distinct sets.
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_val_datasets = random_split(all_files, [train_size, val_size], generator=generator)
    
    if split == "train":
        current_data = train_val_datasets[0].dataset[train_val_datasets[0].indices]
    elif split == "val":
        current_data = train_val_datasets[1].dataset[train_val_datasets[1].indices]
    else:
        raise ValueError(f"Invalid split name: {split}. Expected 'train' or 'val'.")

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

    current_transforms = train_transforms if split == "train" else val_transforms

    # Create MONAI Dataset
    ds = MonaiDataset(data=current_data, transform=current_transforms)

    # Create DataLoader
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=config.get("local", {}).get("num_workers", 4),
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return dataloader


def train_step(model: torch.nn.Module, batch: dict, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    """
    device = next(model.parameters()).device # Get model's device
    inputs, labels = batch["img"].to(device), batch["seg"].to(device)

    # The original script uses DiceLoss with sigmoid=True
    loss_function = monai.losses.DiceLoss(sigmoid=True)

    outputs = model(inputs)
    loss = loss_function(outputs, labels)

    return loss
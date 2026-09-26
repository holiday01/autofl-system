"""
Auto-generated FL client module.
Original script: MONAI 3D Classification example.

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
from monai.data import ImageDataset, DataLoader, Dataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
import monai.networks.nets


# --- Helper for data lists (mimicking original script's hardcoded lists) ---
def _get_ixi_data_lists(data_path: str):
    """
    Helper to get the hardcoded IXI image files and labels from the original script.
    In a real FL scenario, this would be replaced by dynamic data discovery
    based on the client's local data_path.
    """
    images_relative = [
        "IXI314-IOP-0889-T1.nii.gz", "IXI249-Guys-1072-T1.nii.gz", "IXI609-HH-2600-T1.nii.gz",
        "IXI173-HH-1590-T1.nii.gz", "IXI020-Guys-0700-T1.nii.gz", "IXI342-Guys-0909-T1.nii.gz",
        "IXI134-Guys-0780-T1.nii.gz", "IXI577-HH-2661-T1.nii.gz", "IXI066-Guys-0731-T1.nii.gz",
        "IXI130-HH-1528-T1.nii.gz", "IXI607-Guys-1097-T1.nii.gz", "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz", "IXI344-Guys-0905-T1.nii.gz", "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz", "IXI253-HH-1694-T1.nii.gz", "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz", "IXI585-Guys-1130-T1.nii.gz",
    ]
    images = [os.path.join(data_path, f) for f in images_relative]
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    return images, labels


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the MONAI DenseNet121 model.
    Config can specify 'model_kwargs' for spatial_dims, in_channels, out_channels.
    """
    kwargs = config.get("model_kwargs", {})
    # Default values from the original script
    spatial_dims = kwargs.get("spatial_dims", 3)
    in_channels = kwargs.get("in_channels", 1)
    out_channels = kwargs.get("out_channels", 2)
    return monai.networks.nets.DenseNet121(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a MONAI DataLoader for the specified split (train or val).
    Config can specify 'data_path', 'batch_size', 'num_workers', 'pin_memory',
    'image_size' for Resize transform.
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 2))  # Default from original script
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory = local.get("pin_memory", True)
    image_size = config.get("image_size", (96, 96, 96))  # Default from original script

    # Data path from config, with a default that matches the original script's structure
    # In a real FL setup, this path would point to the client's local data.
    data_path = config.get("data_path", os.path.join(".", "workspace", "data", "medical", "ixi", "IXI-T1"))

    # Get the hardcoded image and label lists from the original script
    # In a real FL scenario, this would be replaced by dynamic data discovery
    # based on the client's local data_path.
    all_images, all_labels = _get_ixi_data_lists(data_path)

    # Define transforms based on split
    if split == "train":
        transforms = Compose([
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize(image_size),
            RandRotate90(),
        ])
        # Original script uses images[:10] for train
        dataset = ImageDataset(image_files=all_images[:10], labels=all_labels[:10], transform=transforms)
        shuffle = True
    elif split == "val":
        transforms = Compose([
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize(image_size),
        ])
        # Original script uses images[-10:] for val
        dataset = ImageDataset(image_files=all_images[-10:], labels=all_labels[-10:], transform=transforms)
        shuffle = False
    else:
        raise ValueError(f"Unsupported split: {split}. Must be 'train' or 'val'.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,  # Not used directly, but part of the contract
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    # MONAI DataLoader typically returns a list/tuple of tensors
    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0].to(device), batch[1].to(device)
    elif isinstance(batch, dict):
        # Handle dict batches if MONAI ever returns them, similar to example
        inputs = batch.get("input", batch.get("x", batch.get("image"))).to(device)
        targets = batch.get("label", batch.get("y", batch.get("target"))).to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    criterion = torch.nn.CrossEntropyLoss()  # Instantiate loss function
    loss = criterion(outputs, targets)
    return loss
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Union

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from monai.networks.nets import DenseNet121


# --- Helper to get device ---
def _get_device(model: torch.nn.Module) -> torch.device:
    """Infer the device from the model's parameters."""
    return next(model.parameters()).device


# --- Synthetic Dataset Class ---
class _SyntheticMONAIDataset(Dataset):
    """
    A synthetic dataset that mimics raw image data (e.g., numpy arrays)
    before MONAI transforms are applied.
    """
    def __init__(self, num_samples: int = 20, raw_img_shape: tuple = (128, 128, 128), num_classes: int = 2):
        self.num_samples = num_samples
        self.raw_img_shape = raw_img_shape
        self.num_classes = num_classes
        
        # Pre-generate some dummy data: raw numpy arrays and integer labels
        # Simulating values that might come from a Nifti file (e.g., 0-255 or arbitrary intensity)
        self.images = [np.random.rand(*raw_img_shape).astype(np.float32) * 255 for _ in range(num_samples)]
        self.labels = [np.random.randint(0, num_classes) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# --- Helper for datasets with indices and transforms ---
class _SubDataset(Dataset):
    """
    A wrapper to apply transforms to a subset of a base dataset.
    This is used for synthetic data where a single base dataset is created,
    and then subsets with specific transforms are derived from it.
    """
    def __init__(self, base_dataset: Dataset, indices: list, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, label = self.base_dataset[original_idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long) # Ensure label is a tensor


# --- build_model ---
def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    
    # Default values from the original script if not provided in config
    spatial_dims = model_kwargs.get("spatial_dims", 3)
    in_channels = model_kwargs.get("in_channels", 1)
    out_channels = model_kwargs.get("out_channels", 2)
    
    model = DenseNet121(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels
    )
    return model


# --- build_dataloader ---
def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    
    # Define transforms (as in original script)
    # These transforms are designed to operate on loaded image data (e.g., numpy arrays)
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    # Original script uses hardcoded images and labels relative to data_path
    base_image_names = [
        "IXI314-IOP-0889-T1.nii.gz", "IXI249-Guys-1072-T1.nii.gz", "IXI609-HH-2600-T1.nii.gz",
        "IXI173-HH-1590-T1.nii.gz", "IXI020-Guys-0700-T1.nii.gz", "IXI342-Guys-0909-T1.nii.gz",
        "IXI134-Guys-0780-T1.nii.gz", "IXI577-HH-2661-T1.nii.gz", "IXI066-Guys-0731-T1.nii.gz",
        "IXI130-HH-1528-T1.nii.gz", "IXI607-Guys-1097-T1.nii.gz", "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz", "IXI344-Guys-0905-T1.nii.gz", "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz", "IXI253-HH-1694-T1.nii.gz", "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz", "IXI585-Guys-1130-T1.nii.gz",
    ]
    
    full_image_paths = [os.path.join(data_path, f) for f in base_image_names]
    full_labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

    base_dataset = None
    num_full_samples = 0
    num_workers = config.get("local", {}).get("num_workers", 2)
    pin_memory = torch.cuda.is_available() and config.get("local", {}).get("pin_memory", True)

    # Check if data files exist
    all_files_exist = True
    if full_image_paths: 
        for f_path in full_image_paths:
            if not os.path.exists(f_path):
                all_files_exist = False
                break
    else: # If base_image_names is empty, no real data can be loaded
        all_files_exist = False

    if not all_files_exist:
        if allow_synthetic_data:
            logging.warning(
                f"Data files not found at {data_path}. Generating synthetic data "
                f"for '{split}' split with batch_size={batch_size}."
            )
            # Define parameters for synthetic data generation
            num_full_samples = config.get("synthetic_data_samples", len(base_image_names) if base_image_names else 20)
            raw_img_shape = config.get("synthetic_raw_img_shape", (128, 128, 128)) 
            num_classes = config.get("model_kwargs", {}).get("out_channels", 2)
            
            base_dataset = _SyntheticMONAIDataset(num_samples=num_full_samples, raw_img_shape=raw_img_shape, num_classes=num_classes)
            
            # For synthetic data, it's often safer/simpler to use 0 workers
            num_workers = config.get("local", {}).get("num_workers", 0) 

        else:
            raise FileNotFoundError(
                f"Data files not found at {data_path}. Set 'allow_synthetic_data: True' in config "
                f"to use synthetic data or provide valid data_path."
            )
    else:
        num_full_samples = len(full_image_paths)


    if num_full_samples < 2:
        raise ValueError(
            f"Dataset has only {num_full_samples} samples. Cannot perform train/val split. "
            "Please ensure enough data is available or adjust synthetic_data_samples."
        )

    # Split indices into train_indices and val_indices using random_split logic
    train_ratio = config.get("train_ratio", 0.8)
    train_size = int(train_ratio * num_full_samples)
    val_size = num_full_samples - train_size
    
    # Ensure at least one sample in each split if possible
    if train_size == 0 and val_size > 0: train_size = 1; val_size -= 1
    if val_size == 0 and train_size > 0: val_size = 1; train_size -= 1
    
    if train_size == 0 and val_size == 0:
        raise ValueError("Could not create train/val splits with given dataset size.")

    # Generate shuffled indices once for reproducibility
    g = torch.Generator().manual_seed(config.get("seed", 42))
    shuffled_indices = torch.randperm(num_full_samples, generator=g).tolist()

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    if all_files_exist:
        # Create ImageDataset for train and val splits directly from subsets of paths/labels
        train_ds = ImageDataset(
            image_files=[full_image_paths[i] for i in train_indices],
            labels=np.array([full_labels[i] for i in train_indices], dtype=np.int64),
            transform=train_transforms
        )
        val_ds = ImageDataset(
            image_files=[full_image_paths[i] for i in val_indices],
            labels=np.array([full_labels[i] for i in val_indices], dtype=np.int64),
            transform=val_transforms
        )
    else: # Synthetic data path
        # Use the _SubDataset wrapper with the base_synthetic_ds to apply transforms
        train_ds = _SubDataset(base_dataset, train_indices, train_transforms)
        val_ds = _SubDataset(base_dataset, val_indices, val_transforms)
    
    if split == "train":
        dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    elif split == "val":
        dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    else:
        raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

    return dataloader


# --- train_step ---
def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    Move tensors to the device of the model parameters.
    """
    # Infer device from model
    device = _get_device(model)

    # Instantiate loss function
    # Original script uses torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.CrossEntropyLoss()

    # Move batch data to model's device
    inputs, labels = batch[0].to(device), batch[1].to(device)

    # Forward pass
    outputs = model(inputs)
    
    # Calculate loss
    loss = loss_function(outputs, labels)

    return loss
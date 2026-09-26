# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split

import monai
from monai.data import ImageDataset, DataLoader # Using MONAI's DataLoader as in original script
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
import monai.networks.nets # To import DenseNet121

# The original script hardcoded image filenames and labels
_DATASET_IMAGES = [
    "IXI314-IOP-0889-T1.nii.gz",
    "IXI249-Guys-1072-T1.nii.gz",
    "IXI609-HH-2600-T1.nii.gz",
    "IXI173-HH-1590-T1.nii.gz",
    "IXI020-Guys-0700-T1.nii.gz",
    "IXI342-Guys-0909-T1.nii.gz",
    "IXI134-Guys-0780-T1.nii.gz",
    "IXI577-HH-2661-T1.nii.gz",
    "IXI066-Guys-0731-T1.nii.gz",
    "IXI130-HH-1528-T1.nii.gz",
    "IXI607-Guys-1097-T1.nii.gz",
    "IXI175-HH-1570-T1.nii.gz",
    "IXI385-HH-2078-T1.nii.gz",
    "IXI344-Guys-0905-T1.nii.gz",
    "IXI409-Guys-0960-T1.nii.gz",
    "IXI584-Guys-1129-T1.nii.gz",
    "IXI253-HH-1694-T1.nii.gz",
    "IXI092-HH-1436-T1.nii.gz",
    "IXI574-IOP-1156-T1.nii.gz",
    "IXI585-Guys-1130-T1.nii.gz",
]
_DATASET_LABELS = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

# Define transforms outside functions to avoid re-creation
_train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
_val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

class SyntheticMedicalDataset(Dataset):
    """
    A synthetic dataset for medical imaging tasks, providing dummy images and labels.
    """
    def __init__(self, num_samples: int, image_shape: Tuple[int, ...], num_classes: int, transform=None):
        self.num_samples = num_samples
        self.image_shape = image_shape # Expected (C, D, H, W)
        self.num_classes = num_classes
        self.transform = transform
        logging.info(f"Initialized SyntheticMedicalDataset with {num_samples} samples, image_shape={image_shape}, num_classes={num_classes}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image tensor, float32, in range [0, 1]
        image = torch.rand(self.image_shape, dtype=torch.float32)
        # Generate a random label
        label = torch.randint(0, self.num_classes, (1,), dtype=torch.long).squeeze(0)

        if self.transform:
            image = self.transform(image)
        return image, label

class SplitTransformDataset(Dataset):
    """
    A wrapper dataset to apply specific transforms to subsets created by random_split.
    It expects the base_raw_dataset's __getitem__ to return raw (e.g., numpy) data
    which the 'transform' will then process.
    """
    def __init__(self, base_raw_dataset: ImageDataset, indices: List[int], transform):
        self.base_raw_dataset = base_raw_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        # ImageDataset with transform=None returns (numpy_array, label_int)
        image, label = self.base_raw_dataset[original_idx] 
        if self.transform:
            image = self.transform(image) # Apply the split-specific transform (e.g., to convert numpy to tensor, apply augmentation)
        return image, label


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the DenseNet121 model.
    """
    model_kwargs = config.get("model_kwargs", {})
    # Default parameters based on the original script
    kwargs = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
    }
    kwargs.update(model_kwargs)
    return monai.networks.nets.DenseNet121(**kwargs)

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    Uses random_split to produce train/val subsets from a single dataset.
    Includes a synthetic data fallback.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    # Using 2 num_workers as in the original script. Can be configured.
    num_workers = config.get("local", {}).get("num_workers", 2)
    # Pin memory if CUDA is available, as in the original script.
    pin_memory = torch.cuda.is_available()

    full_image_paths = [os.path.join(data_path, f) for f in _DATASET_IMAGES]
    full_labels = _DATASET_LABELS

    # Check if real data exists
    data_available = True
    if not os.path.isdir(data_path):
        data_available = False
        logging.warning(f"Data directory '{data_path}' not found at '{os.path.abspath(data_path)}'.")
    else:
        for f in full_image_paths:
            if not os.path.exists(f):
                data_available = False
                logging.warning(f"Image file '{f}' not found in '{data_path}'.")
                break

    if not data_available:
        if not allow_synthetic_data:
            raise FileNotFoundError(
                f"Real data not found at '{os.path.abspath(data_path)}' and 'allow_synthetic_data' is False. "
                "Set 'allow_synthetic_data' to True in the client config to use synthetic data."
            )
        else:
            logging.info(f"Using synthetic data for '{split}' split (real data not found or incomplete).")
            num_synthetic_samples = len(_DATASET_IMAGES) # Match original dataset size for synthetic data
            image_shape = (1, 96, 96, 96) # Based on Resize((96, 96, 96)) and EnsureChannelFirst
            num_classes = 2

            # For synthetic data, we apply the training transforms, as it's primarily for testing.
            # No need for separate train/val transforms for synthetic data as its purpose is just to provide data.
            # Using _train_transforms ensures it produces tensors of expected shape.
            dataset = SyntheticMedicalDataset(
                num_samples=num_synthetic_samples,
                image_shape=image_shape,
                num_classes=num_classes,
                transform=_train_transforms
            )
    else:
        # Use real data
        logging.info(f"Using real data from '{os.path.abspath(data_path)}' for '{split}' split.")

        # Create a base ImageDataset without any transforms initially.
        # This will make its __getitem__ return raw numpy arrays.
        base_dataset_raw = ImageDataset(image_files=full_image_paths, labels=full_labels, transform=None)

        # Split into train and validation indices using random_split
        num_total = len(base_dataset_raw)
        if num_total < 2:
            # If only 0 or 1 sample, it's problematic for splitting
            raise ValueError(f"Dataset too small ({num_total} samples) to split into train/val. Need at least 2 samples.")

        train_size = int(0.8 * num_total) # Default 80/20 split
        val_size = num_total - train_size

        # Ensure that splits have at least one sample if possible
        if train_size == 0 and num_total > 0: train_size = 1
        if val_size == 0 and num_total - train_size > 0: val_size = num_total - train_size
        # Readjust train_size if val_size got updated and total needs to be maintained
        if train_size + val_size != num_total:
             train_size = num_total - val_size

        # Check if splits are empty for the requested split type
        if train_size <= 0 and split == "train":
            raise ValueError("Training split is empty after random_split. Please provide more data or adjust split ratio.")
        if val_size <= 0 and split == "val":
            raise ValueError("Validation split is empty after random_split. Please provide more data or adjust split ratio.")


        g = torch.Generator().manual_seed(config.get("seed", 42)) # For reproducible random_split
        
        # random_split can split a list of indices, returning Subsets of these indices
        all_indices = list(range(num_total))
        train_indices_subset, val_indices_subset = random_split(all_indices, [train_size, val_size], generator=g)

        if split == "train":
            dataset = SplitTransformDataset(base_dataset_raw, train_indices_subset.indices, _train_transforms)
        elif split == "val":
            dataset = SplitTransformDataset(base_dataset_raw, val_indices_subset.indices, _val_transforms)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    # Create DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"), # Only shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def train_step(model: torch.nn.Module, batch: Tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    Move tensors to the device of the model parameters.
    """
    inputs, labels = batch
    
    # Move tensors to the device of the model parameters
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Perform forward pass
    outputs = model(inputs)

    # Calculate loss. CrossEntropyLoss is used in the original script.
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(outputs, labels)

    # Return the loss tensor with grad attached
    return loss
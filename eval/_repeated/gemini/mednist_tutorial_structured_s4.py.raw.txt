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

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import monai
from monai.data import ImageDataset, DataLoader as MonaiDataLoader # Use alias to avoid conflict with torch.utils.data.DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
import monai.networks.nets


# Custom Dataset for synthetic data, mimicking MONAI ImageDataset output
class SyntheticImageDataset(Dataset):
    """
    A synthetic dataset that generates random 3D images and binary labels.
    Used as a fallback when real data is unavailable.
    """
    def __init__(self, num_samples: int, transform=None, raw_image_shape=(128, 128, 128)):
        self.num_samples = num_samples
        self.transform = transform
        self.raw_image_shape = raw_image_shape
        # Generate random binary labels (0 or 1 for 2 classes)
        self.labels = torch.randint(0, 2, (num_samples,), dtype=torch.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Generate a synthetic 3D image volume as float32
        synthetic_image = torch.randn(self.raw_image_shape, dtype=torch.float32)
        label = self.labels[idx].item()

        # MONAI transforms are designed to work with numpy arrays or dictionaries by default.
        # Convert tensor to numpy array before applying transforms if any are provided.
        if self.transform:
            synthetic_image_np = synthetic_image.numpy()
            transformed_image = self.transform(synthetic_image_np)
            return transformed_image, label
        
        # If no transform, return the raw tensor and label
        return synthetic_image, label

# Wrapper for torch.utils.data.Subset to apply transforms after splitting
class TransformedSubset(Dataset):
    """
    A wrapper around torch.utils.data.Subset to apply transforms
    to items retrieved from the subset. This is necessary when using
    torch.utils.data.random_split which returns Subset objects.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int):
        # ImageDataset with transform=None returns (numpy_array, int_label)
        # SyntheticImageDataset with transform=None returns (torch_tensor, int_label)
        image, label = self.subset[index]
        
        # Ensure image is a numpy array before applying MONAI transforms,
        # as many MONAI transforms expect numpy or dict inputs.
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    # Default values from the original script
    spatial_dims = model_kwargs.get("spatial_dims", 3)
    in_channels = model_kwargs.get("in_channels", 1)
    out_channels = model_kwargs.get("out_channels", 2)
    
    model = monai.networks.nets.DenseNet121(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> MonaiDataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    
    # num_workers and pin_memory from original script
    num_workers = config.get("local", {}).get("num_workers", 2) 
    pin_memory = torch.cuda.is_available()

    # Define transforms
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    # Original data definition from the script for IXI dataset
    original_images_relative = [
        "IXI314-IOP-0889-T1.nii.gz", "IXI249-Guys-1072-T1.nii.gz", "IXI609-HH-2600-T1.nii.gz",
        "IXI173-HH-1590-T1.nii.gz", "IXI020-Guys-0700-T1.nii.gz", "IXI342-Guys-0909-T1.nii.gz",
        "IXI134-Guys-0780-T1.nii.gz", "IXI577-HH-2661-T1.nii.gz", "IXI066-Guys-0731-T1.nii.gz",
        "IXI130-HH-1528-T1.nii.gz", "IXI607-Guys-1097-T1.nii.gz", "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz", "IXI344-Guys-0905-T1.nii.gz", "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz", "IXI253-HH-1694-T1.nii.gz", "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz", "IXI585-Guys-1130-T1.nii.gz",
    ]
    all_labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

    # Construct full paths to image files based on config.data_path.
    # We assume `data_path` from config is the base directory containing the image files.
    all_image_files = [os.path.join(data_path, f) for f in original_images_relative]

    # Check if all specified real data files actually exist
    all_files_exist = all(os.path.exists(f) for f in all_image_files)

    if not all_files_exist and not allow_synthetic_data:
        # If real data is missing and synthetic data is not allowed, raise an error
        raise FileNotFoundError(
            f"Image files not found in '{data_path}'. "
            f"For example, '{all_image_files[0]}' does not exist. "
            f"Please ensure your data_path is correct or set 'allow_synthetic_data: True' "
            f"in your client config to use synthetic data for testing purposes."
        )
    
    if allow_synthetic_data and not all_files_exist:
        print("WARNING: Using synthetic data. Real data not found or not accessible.")
        num_samples = len(all_labels) # Match the number of samples in the original dataset
        full_dataset_raw = SyntheticImageDataset(num_samples, transform=None)
    else:
        # Load real data. Using transform=None initially; transforms applied via TransformedSubset
        full_dataset_raw = ImageDataset(image_files=all_image_files, labels=all_labels, transform=None)

    # Use random_split to produce train/val subsets from a single dataset
    num_total_samples = len(full_dataset_raw)
    # Split 50/50 for train/val, matching the original script's effective split size (10 train, 10 val)
    train_len = num_total_samples // 2
    val_len = num_total_samples - train_len
    lengths = [train_len, val_len]

    # Use a fixed generator seed for reproducibility of the split
    train_subset_raw, val_subset_raw = random_split(full_dataset_raw, lengths, generator=torch.Generator().manual_seed(42))

    if split == "train":
        dataset = TransformedSubset(train_subset_raw, transform=train_transforms)
    elif split == "val":
        dataset = TransformedSubset(val_subset_raw, transform=val_transforms)
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    # The original script used monai.data.DataLoader
    return MonaiDataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == "train"), # Only shuffle training data
        num_workers=num_workers, 
        pin_memory=pin_memory
    )


def train_step(model: torch.nn.Module, batch, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    """
    inputs, labels = batch[0], batch[1]

    # Move tensors to the device of the model parameters
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Perform forward pass
    outputs = model(inputs)

    # Calculate loss. Original script uses CrossEntropyLoss.
    # Instantiate loss function here for train_step, as it's a fixed part of the training logic.
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(outputs, labels)

    return loss
"""
Auto-generated FL client module.
Original script: monai_classification_3d_array.py

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
import logging

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from monai.networks.nets import DenseNet121


# Configure logging for better feedback
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class _SyntheticMonaiImageDataset(Dataset):
    """
    A synthetic dataset to mimic the output of MONAI ImageDataset after transforms.
    Generates random tensors matching the expected output shape and applies transforms.
    """
    def __init__(self, n: int, labels: np.ndarray, transform: Compose, output_shape: tuple = (1, 96, 96, 96)):
        if len(labels) != n:
            raise ValueError(f"Number of labels ({len(labels)}) must match n ({n}).")
        self.n = n
        self.labels = labels
        self.transform = transform
        self.output_shape = output_shape
        # Pre-generate 'raw' data in the expected input format for the first transform
        # (which is ScaleIntensity, typically expecting numpy array (H, W, D)).
        self._raw_images = [np.random.rand(*output_shape[1:]).astype(np.float32) for _ in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Apply the full transform pipeline
        img_tensor = self.transform(self._raw_images[idx])
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label_tensor


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the MONAI DenseNet121 model.

    Args:
        config: A dictionary containing configuration parameters.
                Expected keys: "model_kwargs" (dict for DenseNet121 constructor).
    Returns:
        An instance of torch.nn.Module (DenseNet121).
    """
    kwargs = config.get("model_kwargs", {})
    # Default parameters based on the original script
    default_model_kwargs = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
    }
    final_kwargs = {**default_model_kwargs, **kwargs}
    return DenseNet121(**final_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a PyTorch DataLoader for the specified data split.

    Args:
        config: A dictionary containing configuration parameters.
                Expected keys:
                    - "local": (dict) client-specific overrides for batch_size, num_workers, pin_memory.
                    - "batch_size": (int) batch size for DataLoader.
                    - "num_workers": (int) number of worker processes for DataLoader.
                    - "pin_memory": (bool) whether to pin memory.
                    - "data_path": (str, optional) path to the root directory of the IXI dataset.
                    - "synthetic_kwargs": (dict, optional) parameters for synthetic dataset if real data
                                          is not available, e.g., "n_total" for total number of samples.
                    - "seed": (int, optional) random seed for data splitting.
        split: A string indicating the data split ("train", "val", or "test").
    Returns:
        An instance of torch.utils.data.DataLoader.
    """
    local_config = config.get("local", {})
    batch_size = local_config.get("batch_size", config.get("batch_size", 2))
    num_workers = local_config.get("num_workers", config.get("num_workers", 2))
    pin_memory = local_config.get("pin_memory", True)
    
    # Target image shape after Resize, before channel first, as per original script
    target_img_hw_d = (96, 96, 96) 
    
    # Define transforms based on split
    if split == "train":
        transforms = Compose([
            ScaleIntensity(),
            EnsureChannelFirst(), # This will convert (H,W,D) numpy to (1,H,W,D) torch.Tensor
            Resize(target_img_hw_d),
            RandRotate90(),
        ])
    elif split == "val" or split == "test":
        transforms = Compose([
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize(target_img_hw_d),
        ])
    else:
        raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'.")

    # Determine dataset source
    data_path = config.get("data_path")
    
    # Hardcoded image filenames and labels from the original script
    full_image_filenames = [
        "IXI314-IOP-0889-T1.nii.gz", "IXI249-Guys-1072-T1.nii.gz", "IXI609-HH-2600-T1.nii.gz",
        "IXI173-HH-1590-T1.nii.gz", "IXI020-Guys-0700-T1.nii.gz", "IXI342-Guys-0909-T1.nii.gz",
        "IXI134-Guys-0780-T1.nii.gz", "IXI577-HH-2661-T1.nii.gz", "IXI066-Guys-0731-T1.nii.gz",
        "IXI130-HH-1528-T1.nii.gz", "IXI607-Guys-1097-T1.nii.gz", "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz", "IXI344-Guys-0905-T1.nii.gz", "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz", "IXI253-HH-1694-T1.nii.gz", "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz", "IXI585-Guys-1130-T1.nii.gz",
    ]
    all_original_labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    
    real_image_paths = []
    
    # Check if a valid data_path is provided and contains all expected files
    if data_path and os.path.isdir(data_path):
        all_files_present = True
        for fname in full_image_filenames:
            full_path = os.path.join(data_path, fname)
            if os.path.exists(full_path):
                real_image_paths.append(full_path)
            else:
                logger.warning(f"Missing data file at {full_path}. Will use synthetic data instead of real data.")
                all_files_present = False
                real_image_paths = [] # Clear collected paths to force synthetic fallback
                break # Exit loop as real data is incomplete

        if all_files_present:
            logger.info(f"Loaded {len(real_image_paths)} real image paths from {data_path}.")
            # The original script uses fixed slices for train/val.
            if split == "train":
                ds = ImageDataset(image_files=real_image_paths[:10], labels=all_original_labels[:10], transform=transforms)
            elif split == "val" or split == "test":
                ds = ImageDataset(image_files=real_image_paths[-10:], labels=all_original_labels[-10:], transform=transforms)
            else:
                raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'.")
        
    if not real_image_paths: # Fallback to synthetic data if no real data was found or complete
        logger.info("Using synthetic data for the MONAI client.")
        n_synthetic_total = config.get("synthetic_kwargs", {}).get("n_total", 20)
        num_classes_for_synthetic = config.get("model_kwargs", {}).get("out_channels", 2)
        
        # Labels for synthetic dataset - simple alternating for demonstration
        synthetic_labels_full = np.array([i % num_classes_for_synthetic for i in range(n_synthetic_total)], dtype=np.int64)
        
        full_synthetic_ds = _SyntheticMonaiImageDataset(
            n=n_synthetic_total,
            labels=synthetic_labels_full,
            transform=transforms,
            output_shape=(1,) + target_img_hw_d # (C, H, W, D) for the final output shape
        )
        
        # For synthetic data, we use random_split. To mimic original 10/10 split if total is 20:
        n_train = 0
        n_val = 0
        if n_synthetic_total == 20:
             n_train = 10
             n_val = 10
        else: # Otherwise, use val_ratio from config
            val_ratio = config.get("val_ratio", 0.1)
            n_val = max(1, int(len(full_synthetic_ds) * val_ratio))
            n_train = len(full_synthetic_ds) - n_val
        
        # Ensure total size matches after splitting
        if n_train + n_val != n_synthetic_total:
             # Adjust due to max(1, ...) if n_synthetic_total is very small
            n_train = n_synthetic_total - n_val

        train_ds, val_ds = torch.utils.data.random_split(
            full_synthetic_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.get("seed", 42)),
        )
        ds = train_ds if split == "train" else val_ds

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"), # Shuffle only for training
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer, # Not used as per contract, but kept for signature consistency
    config: dict,
) -> torch.Tensor:
    """
    Performs one forward pass and calculates the loss.

    Args:
        model: The PyTorch model to train.
        batch: A tuple or list containing (inputs, targets).
        optimizer: The optimizer (not used directly in this function per contract).
        config: A dictionary containing configuration parameters (not used for loss here).

    Returns:
        A scalar `torch.Tensor` representing the loss, with `grad_fn` attached.
    """
    # Determine the device of the model parameters to move batch to
    device = next(model.parameters()).device
    
    # Batch structure from MONAI ImageDataset + DataLoader is typically (image_tensor, label_tensor)
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        inputs, targets = batch[0], batch[1]
    else:
        # Fallback for more general batch structures if needed, or raise error
        raise TypeError(f"Unsupported batch type or structure: {type(batch)}. Expected tuple/list of length 2 (inputs, targets).")

    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    
    # Loss function from the original script (CrossEntropyLoss)
    # Instantiate it here for simplicity. Could also be passed via config.
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(outputs, targets)
    return loss
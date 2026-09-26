import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset

import monai
from monai.data import create_test_image_3d, list_data_collate

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
)
from monai.losses import DiceLoss

# Configure logging for the client module
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the MONAI UNet model.
    """
    model_kwargs = config.get("model_kwargs", {})
    
    # Default parameters based on the original script
    spatial_dims = model_kwargs.get("spatial_dims", 3)
    in_channels = model_kwargs.get("in_channels", 1)
    out_channels = model_kwargs.get("out_channels", 1)
    channels = model_kwargs.get("channels", (16, 32, 64, 128, 256))
    strides = model_kwargs.get("strides", (2, 2, 2, 2))
    num_res_units = model_kwargs.get("num_res_units", 2)

    model = monai.networks.nets.UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
    )
    return model

class FLMONAIDataset(monai.data.Dataset):
    """
    A MONAI Dataset wrapper that can hold a reference to a TemporaryDirectory
    object, ensuring it persists as long as the dataset is alive. This is
    crucial for synthetic data generated into a temporary directory.
    """
    def __init__(self, data, transform, temp_dir_obj=None):
        super().__init__(data, transform)
        self._temp_dir_obj = temp_dir_obj # Keep reference to tempdir to prevent early cleanup

    def cleanup_temp_dir(self):
        """Manually clean up the temporary directory if it was created."""
        if self._temp_dir_obj:
            logging.info(f"Cleaning up synthetic data temporary directory: {self._temp_dir_obj.name}")
            self._temp_dir_obj.cleanup()
            self._temp_dir_obj = None

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    Supports synthetic data fallback gated by config.
    """
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    train_val_split_ratio = config.get("train_val_split_ratio", 0.8) # Default 80/20 split
    num_workers = config.get("local", {}).get("num_workers", 4)
    pin_memory = torch.cuda.is_available()

    # Define transforms based on the original script
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

    all_files = []
    real_data_found = False
    
    # Check for real data at the specified data_path
    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if images and segs:
        if len(images) == len(segs) and len(images) > 0:
            all_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
            real_data_found = True
            logging.info(f"Found {len(all_files)} real data samples at {data_path}.")
        else:
            logging.warning(f"Mismatched or empty real data found at {data_path} (Images: {len(images)}, Segments: {len(segs)}).")

    temp_synth_dir_obj = None
    if not real_data_found:
        if allow_synthetic_data:
            logging.warning(f"No valid real data found at '{data_path}'. Generating synthetic data into a temporary directory.")
            # Create a TemporaryDirectory object and store its reference to prevent early cleanup
            temp_synth_dir_obj = tempfile.TemporaryDirectory()
            synth_data_path = temp_synth_dir_obj.name
            
            num_synthetic_samples = 40 # Based on the original script's total sample count
            for i in range(num_synthetic_samples):
                im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
                nib.save(nib.Nifti1Image(im, np.eye(4)), os.path.join(synth_data_path, f"img{i:d}.nii.gz"))
                nib.save(nib.Nifti1Image(seg, np.eye(4)), os.path.join(synth_data_path, f"seg{i:d}.nii.gz"))
            
            images = sorted(glob(os.path.join(synth_data_path, "img*.nii.gz")))
            segs = sorted(glob(os.path.join(synth_data_path, "seg*.nii.gz")))
            all_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
            logging.info(f"Generated {len(all_files)} synthetic data samples in {synth_data_path}.")
        else:
            raise FileNotFoundError(
                f"No real data found at '{data_path}' and 'allow_synthetic_data' is False. "
                "Cannot create DataLoader without data."
            )

    if not all_files:
        raise RuntimeError("No data files (real or synthetic) could be prepared for the DataLoader.")

    # Determine which transform to use based on the split
    dataset_transform = train_transforms if split == "train" else val_transforms
    
    # Create the full dataset, passing the temp_dir_obj if synthetic data was created
    full_dataset = FLMONAIDataset(data=all_files, transform=dataset_transform, temp_dir_obj=temp_synth_dir_obj)

    num_total_samples = len(full_dataset)
    if num_total_samples == 0:
        logging.warning("No samples in dataset after preparation. Returning empty DataLoader.")
        return DataLoader([], batch_size=batch_size, collate_fn=list_data_collate)

    # Calculate train/val split sizes
    num_train_samples = int(num_total_samples * train_val_split_ratio)
    num_val_samples = num_total_samples - num_train_samples

    # Ensure at least one sample in each split if possible, if ratio leads to zero
    if num_total_samples > 0:
        if num_train_samples == 0 and num_val_samples > 0:
            num_train_samples = 1
            num_val_samples = num_total_samples - 1
            logging.warning(f"Adjusted split: train has {num_train_samples}, val has {num_val_samples}.")
        elif num_val_samples == 0 and num_train_samples > 0:
            num_val_samples = 1
            num_train_samples = num_total_samples - 1
            logging.warning(f"Adjusted split: train has {num_train_samples}, val has {num_val_samples}.")
        elif num_train_samples == 0 and num_val_samples == 0 and num_total_samples > 0: # Should not happen with total_samples > 0
            num_train_samples = num_total_samples # put all in train if no specific split
            logging.warning(f"Adjusted split: all {num_total_samples} samples put into train, val is 0.")


    # Use a fixed seed for random_split to ensure reproducibility across calls
    generator = torch.Generator().manual_seed(config.get("seed", 42)) 
    train_subset, val_subset = random_split(full_dataset, [num_train_samples, num_val_samples], generator=generator)

    # Return the DataLoader for the requested split
    if split == "train":
        if len(train_subset) == 0:
            logging.warning("Training subset is empty. Returning empty DataLoader for 'train' split.")
            return DataLoader([], batch_size=batch_size, collate_fn=list_data_collate)
        return DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True, # Shuffle training data
            num_workers=num_workers,
            collate_fn=list_data_collate,
            pin_memory=pin_memory,
        )
    elif split == "val":
        if len(val_subset) == 0:
            logging.warning("Validation subset is empty. Returning empty DataLoader for 'val' split.")
            return DataLoader([], batch_size=batch_size, collate_fn=list_data_collate)
        return DataLoader(
            val_subset,
            batch_size=batch_size, # Use configurable batch_size as per prompt, even if original used 1 for val_loader
            shuffle=False,
            num_workers=num_workers,
            collate_fn=list_data_collate,
            pin_memory=pin_memory,
        )
    else:
        raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

# Instantiate DiceLoss once globally for efficiency
_loss_function = DiceLoss(sigmoid=True)

def train_step(model: torch.nn.Module, batch: dict, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass, return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    """
    # Determine device from model parameters
    # Assumes the model is already on the correct device by the FL runtime
    device = next(model.parameters()).device

    inputs, labels = batch["img"].to(device), batch["seg"].to(device)

    # The FL runtime is responsible for optimizer.zero_grad(), loss.backward(), and optimizer.step()
    outputs = model(inputs)
    loss = _loss_function(outputs, labels)

    return loss
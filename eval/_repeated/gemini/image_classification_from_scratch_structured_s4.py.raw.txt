import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import random
import logging

# Set up logging for client
logger = logging.getLogger(__name__)

# --- Utility for filtering corrupted images (from original script) ---
def filter_corrupted_images(data_path):
    num_skipped = 0
    corrupted_files = []
    
    # Check if data_path exists
    if not os.path.isdir(data_path):
        logger.warning(f"Data path '{data_path}' not found for filtering corrupted images.")
        return # Skip filtering if path doesn't exist

    # The original script assumes "Cat" and "Dog" subfolders.
    # ImageFolder expects this structure.
    class_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    if not class_folders:
        logger.warning(f"No class subfolders found in '{data_path}'. Filtering may not be effective.")
        return

    for folder_name in class_folders:
        folder_full_path = os.path.join(data_path, folder_name)
        
        for fname in os.listdir(folder_full_path):
            fpath = os.path.join(folder_full_path, fname)
            if not os.path.isfile(fpath): 
                continue # Skip if it's not a file (e.g., a subdirectory)
            try:
                with open(fpath, "rb") as fobj:
                    # Peek at the first 10 bytes to check for JFIF marker.
                    # This check is specific to JPEG files that contain EXIF metadata.
                    is_jfif = b"JFIF" in fobj.peek(10)
            except Exception as e:
                # Catch exceptions like PermissionError, IsADirectoryError, etc.
                logger.debug(f"Error checking file {fpath}: {e}")
                is_jfif = False # Treat as corrupted or unreadable

            if not is_jfif:
                num_skipped += 1
                corrupted_files.append(fpath)

    if num_skipped > 0:
        logger.info(f"Identified {num_skipped} corrupted images. Deleting them...")
        for fpath in corrupted_files:
            try:
                os.remove(fpath)
            except OSError as e:
                logger.error(f"Error deleting corrupted file {fpath}: {e}")
        logger.info(f"Deleted {num_skipped} images.")
    else:
        logger.info("No corrupted images found.")

# --- PyTorch Model (KerasXceptionLike) ---
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class KerasXceptionLike(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # input_shape from Keras is (H, W, C), e.g., (180, 180, 3)
        self.input_channels = input_shape[2] # 3 for RGB images

        # Entry block
        self.conv1 = nn.Conv2d(self.input_channels, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleList()
        current_channels = 128 # Output channels of the entry block (after conv1)

        for i, size in enumerate([256, 512, 728]):
            # Main block sequence
            block_sequence = nn.Sequential(
                nn.ReLU(inplace=True),
                SeparableConv2d(current_channels, size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
                SeparableConv2d(size, size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(size),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.blocks.append(block_sequence)

            # Residual projection for the block (applied to `previous_block_activation`)
            residual_proj = nn.Conv2d(current_channels, size, kernel_size=1, stride=2, padding=0, bias=False)
            self.blocks.append(residual_proj) # Appending projection after each block_sequence
            
            current_channels = size # Update current_channels for the next iteration

        # Final block before global pooling
        self.final_separable_conv = SeparableConv2d(current_channels, 1024, kernel_size=3, padding=1, bias=False)
        self.final_bn = nn.BatchNorm2d(1024)
        self.final_relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)

        out_units = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(1024, out_units)

    def forward(self, x):
        # Assuming input x is (batch, C, H, W)
        
        # Rescaling (equivalent to Keras layers.Rescaling(1.0 / 255))
        # This means input pixels are expected in [0, 255] range
        x = x / 255.0

        # Entry block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        previous_block_activation = x # Output of entry block (B, 128, H/2, W/2)

        # Intermediate blocks with residuals
        block_step = 0
        for i, size in enumerate([256, 512, 728]):
            # Main block sequence (convs, BN, ReLU, MaxPool)
            x = self.blocks[block_step](x)
            block_step += 1

            # Residual path: Conv2D(size, 1, strides=2) applied to `previous_block_activation`
            residual = self.blocks[block_step](previous_block_activation)
            block_step += 1
            
            x = x + residual # Add back residual
            previous_block_activation = x # Set aside next residual (output of current block)

        # Final block before global pooling
        x = self.final_separable_conv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1) # Flatten to (batch, features)

        x = self.dropout(x)
        x = self.classifier(x)
        return x

# --- Custom Dataset Wrapper for applying transforms after Subset creation ---
class TransformWrapper(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # ImageFolder returns a PIL Image
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

# --- FL Client Module Functions ---

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    image_size = model_kwargs.get("image_size", (180, 180))
    num_classes = model_kwargs.get("num_classes", 2)
    
    # Keras input_shape is (H, W, C), so (image_size[0], image_size[1], 3)
    input_shape = (image_size[0], image_size[1], 3)
    
    return KerasXceptionLike(input_shape=input_shape, num_classes=num_classes)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    seed = config.get("seed", 1337) # Use same seed as original script for split consistency

    image_size = config.get("model_kwargs", {}).get("image_size", (180, 180))
    num_classes = config.get("model_kwargs", {}).get("num_classes", 2) # used for synthetic data

    validation_split_ratio = config.get("validation_split", 0.2)

    # Synthetic data fallback
    if allow_synthetic_data:
        logger.warning("Using synthetic data.")
        C, H, W = 3, image_size[0], image_size[1]
        
        class SyntheticDataset(Dataset):
            def __init__(self, num_samples, num_classes, C, H, W, seed):
                self.num_samples = num_samples
                self.num_classes = num_classes
                self.C, self.H, self.W = C, H, W
                self.generator = torch.Generator().manual_seed(seed)

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # Data in [0, 255] range as float tensor to match Keras model's Rescaling behavior
                # Using torch.rand and scaling to simulate images in [0, 255]
                image = torch.rand(self.C, self.H, self.W, generator=self.generator) * 255.0
                # Labels for binary classification (0 or 1), convert to float for BCEWithLogitsLoss
                label = torch.randint(0, self.num_classes, (1,), dtype=torch.float, generator=self.generator)
                return image, label
        
        total_samples = config.get("synthetic_data_samples", 1000) 
        dataset = SyntheticDataset(total_samples, num_classes, C, H, W, seed)
        
        generator = torch.Generator().manual_seed(seed)
        val_samples = int(total_samples * validation_split_ratio)
        train_samples = total_samples - val_samples
        train_dataset, val_dataset = random_split(dataset, [train_samples, val_samples], generator=generator)

        if split == "train":
            dataset_to_load = train_dataset
        elif split == "val":
            dataset_to_load = val_dataset
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

        return DataLoader(dataset_to_load, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)


    # Real data loading
    pet_images_path = os.path.join(data_path, "PetImages")
    if not os.path.isdir(pet_images_path):
        raise FileNotFoundError(
            f"Dataset not found at '{pet_images_path}'. "
            "Please ensure 'PetImages' directory exists with 'Cat' and 'Dog' subfolders, "
            "or enable 'allow_synthetic_data'."
        )
    
    # Run filtering of corrupted images
    filter_corrupted_images(pet_images_path)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=int(0.1 * 360)), 
        transforms.PILToTensor(), # Converts PIL Image to uint8 Tensor (C, H, W)
        transforms.ConvertImageDtype(torch.float), # Converts uint8 to float, keeping values [0, 255]
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.PILToTensor(), # Converts PIL Image to uint8 Tensor (C, H, W)
        transforms.ConvertImageDtype(torch.float), # Converts uint8 to float, keeping values [0, 255]
    ])

    # Create the full dataset WITHOUT transforms first
    full_dataset_no_transform = ImageFolder(root=pet_images_path, transform=None)

    # Split the dataset using a fixed seed
    total_samples = len(full_dataset_no_transform)
    generator = torch.Generator().manual_seed(seed)
    
    val_samples = int(total_samples * validation_split_ratio)
    train_samples = total_samples - val_samples

    if train_samples <= 0 and split == "train":
        raise ValueError(f"Not enough training samples ({train_samples}) after split. Adjust validation_split or dataset size.")
    if val_samples <= 0 and split == "val":
        raise ValueError(f"Not enough validation samples ({val_samples}) after split. Adjust validation_split or dataset size.")

    train_subset, val_subset = random_split(full_dataset_no_transform, [train_samples, val_samples], generator=generator)

    if split == "train":
        dataset_to_load = TransformWrapper(train_subset, train_transform)
        shuffle = True
    elif split == "val":
        dataset_to_load = TransformWrapper(val_subset, val_transform)
        shuffle = False
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    num_workers = config.get("local", {}).get("num_workers", 0)
    pin_memory = torch.cuda.is_available() # Enable if GPU is present

    return DataLoader(
        dataset_to_load,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    """
    # Get device from model parameters
    device = next(model.parameters()).device

    images, labels = batch
    images = images.to(device)
    
    # Original Keras model uses BinaryCrossentropy(from_logits=True).
    # This means the model output is logits, and labels should be float (0.0 or 1.0).
    # ImageFolder gives integer labels (0 or 1), so convert to float and unsqueeze for BCEWithLogitsLoss.
    labels = labels.float().unsqueeze(1).to(device) 

    # Forward pass
    outputs = model(images)
    
    # Calculate loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(outputs, labels)

    return loss
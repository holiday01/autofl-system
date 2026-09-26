import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
import io # Used in is_valid_image_file for fobj.peek

# --- Model Definition (Keras to PyTorch Conversion) ---

# Helper for SeparableConv2D, translating Keras's SeparableConv2D
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        # Depthwise convolution: each input channel is convolved with its own set of filters
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        # Pointwise convolution: 1x1 convolution to combine the output channels of the depthwise conv
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# The main model, translated from the Keras `make_model` function
class CatsVsDogsModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # input_shape for PyTorch is (C, H, W). Original Keras: (H, W, C), e.g., (180, 180, 3).
        assert len(input_shape) == 3, "Input shape must be (channels, height, width)"
        in_channels = input_shape[0]

        # Rescaling layer: Keras model expects float32 [0, 255] input, then scales to [0, 1].
        # Our PyTorch dataloader will provide uint8 [0, 255] tensors (via PILToTensor).
        # So, we convert to float and then scale by 1/255.0.
        self.rescale = transforms.Lambda(lambda x: x.float() / 255.0)

        # Entry block
        # Keras Conv2D(..., padding="same") with kernel_size=3, stride=2 translates to padding=1.
        self.entry_conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.entry_bn1 = nn.BatchNorm2d(128)
        self.entry_relu1 = nn.ReLU()

        self.blocks = nn.ModuleList()
        current_channels = 128
        # Define the sizes for the Xception-like blocks
        for i, size in enumerate([256, 512, 728]):
            # SeparableConv2D block structure
            block = nn.Sequential(
                nn.ReLU(),
                SeparableConv2d(current_channels, size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(size),

                nn.ReLU(),
                SeparableConv2d(size, size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(size),

                nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Keras MaxPooling2D(3, strides=2, padding="same")
            )
            self.blocks.append(block)

            # Residual projection (1x1 Conv2D with stride 2 to match dimensions after pooling)
            # Keras Conv2D(size, 1, strides=2, padding="same") for 1x1 conv with stride 2 corresponds to padding=0
            # in PyTorch to achieve output size of ceil(input_size / 2).
            self.blocks.append(
                nn.Conv2d(current_channels, size, kernel_size=1, stride=2, padding=0, bias=False)
            )
            current_channels = size

        # Final block
        self.final_separable_conv = SeparableConv2d(current_channels, 1024, kernel_size=3, padding=1, bias=False)
        self.final_bn = nn.BatchNorm2d(1024)
        self.final_relu = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # GlobalAveragePooling2D
        self.dropout = nn.Dropout(0.25)

        # Output layer
        if num_classes == 2:
            units = 1 # Binary classification, output a single logit (for BCEWithLogitsLoss)
        else:
            units = num_classes # Multiclass classification
        self.classifier = nn.Linear(1024, units) # Input features from GlobalAveragePooling2D is 1024

    def forward(self, x):
        # x is expected to be a `torch.Tensor` of shape `(B, C, H, W)` with `uint8` values `[0, 255]`
        x = self.rescale(x) # Scale to `[0, 1]`

        # Entry block
        x = self.entry_conv1(x)
        x = self.entry_bn1(x)
        x = self.entry_relu1(x)

        previous_block_activation = x # Set aside residual connection for the first block

        # Iterate through block and then its corresponding residual projection
        for i in range(0, len(self.blocks), 2):
            block_module = self.blocks[i]
            residual_module = self.blocks[i+1] # This is the residual projection for the *previous* stage

            x = block_module(x)

            residual = residual_module(previous_block_activation)
            x = x + residual # Add back residual
            previous_block_activation = x # Set aside current output as the next residual

        # Final block
        x = self.final_separable_conv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1) # Flatten (B, 1024, 1, 1) to (B, 1024)
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs

def build_model(config: dict) -> torch.nn.Module:
    # Default input shape (channels, height, width) for PyTorch
    # Based on Keras image_size=(180, 180) and 3 channels
    input_shape = config.get("model_kwargs", {}).get("input_shape", (3, 180, 180))
    # Default to 2 classes for Cats vs Dogs problem
    num_classes = config.get("model_kwargs", {}).get("num_classes", 2)
    return CatsVsDogsModel(input_shape, num_classes)


# --- DataLoader Definition ---

# Custom function to validate image files, including JFIF check from original script
def is_valid_image_file(filepath):
    try:
        # Check for "JFIF" marker in the first 10 bytes (JPEG standard marker)
        with open(filepath, "rb") as fobj:
            header = fobj.peek(10) # Read without advancing file pointer
            if b"JFIF" not in header:
                return False
        
        # Attempt to open with PIL to catch other generic image corruption issues
        with Image.open(filepath) as img:
            img.verify() # Verify file integrity
        return True
    except (UnidentifiedImageError, IOError, SyntaxError):
        # PIL raises these errors for corrupted or unreadable images
        return False
    except Exception:
        # Catch any other unexpected errors during file access or parsing
        return False

# Custom ImageFolder that uses our specific image validation function
class FilteredImageFolder(ImageFolder):
    """
    An ImageFolder that filters out corrupted images based on JFIF check
    and general PIL verification, mimicking the original Keras script's pre-filtering.
    """
    def __init__(self, root, transform=None, target_transform=None):
        # Pass our custom `is_valid_image_file` function to the ImageFolder constructor
        super().__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_image_file)


# Wrapper Dataset to apply transforms to subsets obtained from random_split
class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    # image_hw for torchvision transforms is (H, W).
    # model_kwargs input_shape is (C, H, W).
    image_hw = config.get("model_kwargs", {}).get("input_shape", (3, 180, 180))[1:] # (H, W) = (180, 180)
    num_classes = config.get("model_kwargs", {}).get("num_classes", 2) # For synthetic data labels

    # Data augmentation transforms for the training split
    train_transforms = transforms.Compose([
        transforms.Resize(image_hw),                       # Resize all images to the target size
        transforms.RandomHorizontalFlip(p=0.5),            # Keras RandomFlip corresponds to 50% probability
        transforms.RandomRotation(degrees=0.1 * 360),      # Keras RandomRotation(0.1) means up to 10% of 360 degrees rotation
        transforms.PILToTensor(),                          # Converts PIL Image to uint8 Tensor (C, H, W) in range [0, 255]
    ])

    # Transforms for validation/test split (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(image_hw),
        transforms.PILToTensor(),                          # Converts PIL Image to uint8 Tensor (C, H, W) in range [0, 255]
    ])

    dataset = None
    try:
        # ImageFolder expects the `root` to contain class subfolders (e.g., `PetImages/Cat`, `PetImages/Dog`).
        # The original script points to "PetImages".
        full_dataset = FilteredImageFolder(root=data_path) 

        if len(full_dataset) == 0:
            raise FileNotFoundError(f"No valid images found in {data_path}. Please check the path, file integrity, and ensure 'Cat'/'Dog' subfolders exist directly under '{data_path}'.")

        # Split the dataset into training and validation subsets.
        # Original script uses 0.2 for validation, meaning 80% train, 20% val.
        val_split_ratio = 0.2
        num_total = len(full_dataset)
        num_val = int(num_total * val_split_ratio)
        num_train = num_total - num_val
        
        # Ensure sum of split sizes equals total in case of rounding
        if num_train + num_val != num_total:
             num_train = num_total - num_val # Allocate any remaining samples to training

        # Ensure consistent split across clients by using a fixed generator seed if provided
        generator = torch.Generator().manual_seed(config.get("seed", 42))
        train_subset, val_subset = random_split(full_dataset, [num_train, num_val], generator=generator)

        # Apply appropriate transforms to the created subsets
        train_dataset = TransformSubset(train_subset, transform=train_transforms)
        val_dataset = TransformSubset(val_subset, transform=val_transforms)

        if split == "train":
            dataset = train_dataset
        elif split == "val":
            dataset = val_dataset
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'train' or 'val'.")

    except Exception as e:
        if allow_synthetic_data:
            print(f"Warning: Could not load real data from {data_path} due to {type(e).__name__}: {e}. Generating synthetic data.")
            # Generate synthetic data if real data loading fails and allowed
            num_samples = config.get("synthetic_data_samples", 1000)
            if split == "val":
                num_samples = max(1, num_samples // 4) # Smaller validation set for synthetic data
            
            # Synthetic images: (num_samples, channels, height, width), uint8, range [0, 255]
            synthetic_images = torch.randint(0, 256, (num_samples, 3, image_hw[0], image_hw[1]), dtype=torch.uint8)
            synthetic_labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
            dataset = TensorDataset(synthetic_images, synthetic_labels)
        else:
            raise FileNotFoundError(
                f"Real dataset not found or corrupted at '{data_path}' and "
                f"synthetic data generation is not allowed (config['allow_synthetic_data'] is False)."
            ) from e

    # DataLoader configuration
    num_workers = config.get("dataloader_kwargs", {}).get("num_workers", 0)
    pin_memory = config.get("dataloader_kwargs", {}).get("pin_memory", False)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"), # Only shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), # Use persistent workers only if num_workers > 0
    )


# --- Training Step Definition ---

def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    # Ensure model parameters are on the correct device (e.g., GPU)
    device = next(model.parameters()).device
    
    images, labels = batch
    images = images.to(device)

    # Prepare labels for the loss function:
    # Keras used BinaryCrossentropy(from_logits=True).
    # For PyTorch `nn.BCEWithLogitsLoss`, labels should be float32 and typically have shape (batch_size, 1).
    if model.classifier.out_features == 1: # Binary classification
        labels = labels.float().unsqueeze(1).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
    else: # Multiclass classification (e.g., if num_classes > 2)
        labels = labels.long().to(device)
        loss_fn = nn.CrossEntropyLoss()

    # Perform a single forward pass
    outputs = model(images)
    
    # Calculate the loss
    loss = loss_fn(outputs, labels)
    
    # Important: Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    return loss
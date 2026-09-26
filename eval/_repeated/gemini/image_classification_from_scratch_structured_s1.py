import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from PIL import Image

# Helper class for Keras-like SeparableConv2D
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, depth_multiplier=1, bias=False):
        super().__init__()
        # Keras SeparableConv2D typically has depth_multiplier=1, so intermediate_channels = in_channels
        intermediate_channels = in_channels * depth_multiplier 
        self.depthwise = nn.Conv2d(in_channels, intermediate_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Model architecture, converted from Keras to PyTorch
class XceptionLikeModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Keras input_shape is (H, W, C), PyTorch expects (C, H, W)
        # We assume input_shape is provided as (H, W, C) from config
        if len(input_shape) == 3:
            in_channels = input_shape[2]
        else:
            raise ValueError(f"input_shape must be (H, W, C), but got {input_shape}")

        # Entry block
        # Keras `Rescaling(1./255)` is applied as the first layer.
        # This implies inputs are [0, 255] float, and output is [0, 1] float.
        # We handle this by dividing by 255.0 in the forward pass.
        self.entry_conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.entry_bn1 = nn.BatchNorm2d(128)
        self.entry_relu1 = nn.ReLU()

        # Intermediate blocks
        self.blocks = nn.ModuleList()
        current_channels = 128

        for size in [256, 512, 728]:
            block = nn.ModuleDict()
            block['relu1'] = nn.ReLU()
            block['sepconv1'] = SeparableConv2d(current_channels, current_channels, kernel_size=3, padding=1, bias=False)
            block['bn1'] = nn.BatchNorm2d(current_channels)

            block['relu2'] = nn.ReLU()
            block['sepconv2'] = SeparableConv2d(current_channels, size, kernel_size=3, padding=1, bias=False)
            block['bn2'] = nn.BatchNorm2d(size)

            block['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Project residual
            block['residual_conv'] = nn.Conv2d(current_channels, size, kernel_size=1, stride=2, bias=False)
            self.blocks.append(block)
            current_channels = size

        # Final block
        self.final_sepconv = SeparableConv2d(current_channels, 1024, kernel_size=3, padding=1, bias=False)
        self.final_bn = nn.BatchNorm2d(1024)
        self.final_relu = nn.ReLU()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)

        # Keras uses 1 unit for binary classification with `from_logits=True` BinaryCrossentropy
        if num_classes == 2:
            self.classifier = nn.Linear(1024, 1)
        else:
            self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Apply rescaling within the model as per Keras script:
        # Keras `image_dataset_from_directory` outputs float32 in [0, 255] by default.
        # `Rescaling(1./255)` maps [0, 255] to [0, 1].
        # Our DataLoader ensures x is float32 in [0, 255] range.
        x = x / 255.0

        # Entry block
        x = self.entry_conv1(x)
        x = self.entry_bn1(x)
        x = self.entry_relu1(x)

        previous_block_activation = x

        # Intermediate blocks
        for block in self.blocks:
            x = block['relu1'](x)
            x = block['sepconv1'](x)
            x = block['bn1'](x)

            x = block['relu2'](x)
            x = block['sepconv2'](x)
            x = block['bn2'](x)

            x = block['maxpool'](x)

            # Project residual
            residual = block['residual_conv'](previous_block_activation)
            x = x + residual
            previous_block_activation = x

        # Final block
        x = self.final_sepconv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1) # Flatten except batch dimension

        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs

# Helper function to filter corrupted images (from original script)
def filter_corrupted_images(data_path):
    print(f"Filtering corrupted images in {data_path}...")
    num_skipped = 0
    # The original script assumes "PetImages" has "Cat" and "Dog" subfolders
    for folder_name in ("Cat", "Dog"):
        folder_full_path = os.path.join(data_path, folder_name)
        if not os.path.isdir(folder_full_path):
            print(f"Warning: Class folder {folder_full_path} not found. Skipping corruption check.")
            continue

        # Get list of files to avoid issues with os.remove modifying the iterator
        file_list = [os.path.join(folder_full_path, fname) for fname in os.listdir(folder_full_path)]

        for fpath in file_list:
            if not os.path.isfile(fpath):
                continue

            try:
                with open(fpath, "rb") as fobj:
                    # Original check: "JFIF" in the first 10 bytes
                    header = fobj.read(10)
                    is_jfif = b"JFIF" in header
            except Exception: # Catch any file I/O errors or permission issues
                is_jfif = False

            if not is_jfif:
                num_skipped += 1
                try:
                    os.remove(fpath)
                    # print(f"Deleted corrupted image: {fpath}") # Uncomment for verbose debugging
                except OSError as e:
                    print(f"Error deleting file {fpath}: {e}")
    if num_skipped > 0:
        print(f"Deleted {num_skipped} images based on JFIF check.")
    else:
        print("No corrupted images found or deleted based on JFIF check.")
    return num_skipped

# Helper Dataset for synthetic data generation
class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples=100, image_size=(180, 180), num_classes=2, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        self.data = []
        for _ in range(num_samples):
            # Generate random images (float, [0, 255]) and labels
            # Images should have 3 channels (RGB)
            image = Image.fromarray(np.uint8(np.random.rand(image_size[0], image_size[1], 3) * 255))
            label = random.randint(0, num_classes - 1)
            self.data.append((image, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Helper Dataset to apply transforms to subsets obtained from random_split
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
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
    # Default image_size for this problem is (180, 180) and 3 channels
    # num_classes is 2 for cats vs dogs
    input_shape = model_kwargs.get("input_shape", (180, 180, 3))
    num_classes = model_kwargs.get("num_classes", 2)
    return XceptionLikeModel(input_shape=input_shape, num_classes=num_classes)

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)
    # image_size is (H, W), extracted from model_kwargs input_shape (H, W, C)
    image_size = config.get("model_kwargs", {}).get("input_shape", (180, 180, 3))[:2]
    num_classes = config.get("model_kwargs", {}).get("num_classes", 2)
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    seed = config.get("seed", 1337) # Use the same seed as original Keras script for reproducibility

    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError(f"image_size must be a tuple (H, W), but got {image_size}")

    # Transforms for training data (includes augmentation)
    # Keras data_augmentation applies RandomFlip and RandomRotation.
    # Keras RandomRotation(0.1) implies an angle from [-0.1*360, 0.1*360] degrees, i.e., [-36, 36].
    # To match the model's internal Rescaling(1./255), images must be [0, 255] float before entering model.
    # ToTensor() converts PIL [0, 255] to float [0, 1]. So, we multiply by 255.0 to get back to [0, 255].
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=36),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0) # Scale back to [0.0, 255.0]
    ])

    # Transforms for validation data (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0) # Scale back to [0.0, 255.0]
    ])

    full_data_dir = os.path.join(data_path, "PetImages")
    
    # Check for real data availability
    if not os.path.exists(full_data_dir):
        if allow_synthetic_data:
            print(f"WARNING: Real data directory '{full_data_dir}' not found. Using synthetic data.")
            # Create a base synthetic dataset with a generic transform, then wrap subsets
            base_synthetic_dataset = SyntheticImageDataset(
                num_samples=200, # A reasonable size for synthetic data
                image_size=image_size,
                num_classes=num_classes,
                transform=None # Apply no transform initially, TransformedSubset will handle
            )
            # Keras `validation_split=0.2` means 80% train, 20% validation
            train_size = int(0.8 * len(base_synthetic_dataset))
            val_size = len(base_synthetic_dataset) - train_size
            
            # Use random_split to create train/val subsets
            train_subset, val_subset = random_split(
                base_synthetic_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            
            # Apply appropriate transforms to the subsets
            train_dataset = TransformedSubset(train_subset, train_transform)
            val_dataset = TransformedSubset(val_subset, val_transform)

        else:
            raise FileNotFoundError(
                f"Data directory '{full_data_dir}' not found. "
                "Set 'allow_synthetic_data: True' in the client config to use synthetic data."
            )
    else:
        # Filter corrupted images as per original script
        filter_corrupted_images(full_data_dir)

        # Load the real dataset using ImageFolder
        # Apply no transform initially to the full dataset, so random_split works on raw images.
        # Transforms will be applied to the subsets later.
        full_dataset = datasets.ImageFolder(root=full_data_dir, transform=None)

        # Split the dataset into train and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_subset, val_subset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        # Wrap subsets with appropriate transforms for train and validation
        train_dataset = TransformedSubset(train_subset, train_transform)
        val_dataset = TransformedSubset(val_subset, val_transform)

    dataloader_args = {
        "batch_size": batch_size,
        "num_workers": config.get("num_workers", 2),
        "pin_memory": True,
        "persistent_workers": True if config.get("num_workers", 2) > 0 else False,
    }

    if split == "train":
        return DataLoader(train_dataset, shuffle=True, **dataloader_args)
    elif split == "val":
        return DataLoader(val_dataset, shuffle=False, **dataloader_args)
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step().
    """
    device = next(model.parameters()).device
    images, labels = batch
    images = images.to(device)
    
    # Keras uses BinaryCrossentropy(from_logits=True) for 2 classes (output unit 1)
    # PyTorch's BCEWithLogitsLoss expects labels to be float for binary classification
    # and match the output shape (e.g., [batch_size, 1]).
    labels = labels.to(device).float().unsqueeze(1)

    # Forward pass
    logits = model(images)

    # Loss calculation
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)

    return loss
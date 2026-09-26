import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from PIL import Image
import os
import numpy as np

# Helper function to filter corrupted JPEG images, as in the original script.
# This modifies the filesystem by deleting corrupted files.
def filter_corrupted_images(dataset_root_path):
    num_skipped = 0
    if not os.path.exists(dataset_root_path):
        return 0

    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(dataset_root_path, folder_name)
        if not os.path.exists(folder_path):
            continue

        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                # Keras uses fobj.peek(10) which doesn't advance pointer.
                # Python open and read advances.
                with open(fpath, "rb") as fobj:
                    header = fobj.read(10)
                    is_jfif = b"JFIF" in header
            except Exception: # Catch potential errors like truncated files, permission issues
                is_jfif = False

            if not is_jfif:
                num_skipped += 1
                try:
                    os.remove(fpath)
                except OSError as e:
                    print(f"Warning: Could not delete corrupted image {fpath}: {e}")
    if num_skipped > 0:
        print(f"Deleted {num_skipped} corrupted images from {dataset_root_path}.")
    return num_skipped

# PyTorch implementation of Keras's SeparableConv2D
class KerasLikeSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', use_bias=True):
        super().__init__()
        # Keras 'same' padding for stride=1, kernel_size=3 is padding=1.
        # Keras 'same' padding for stride=2, kernel_size=3 results in output ceil(input/2).
        # PyTorch padding=1 for kernel_size=3, stride=2 gives output floor((input + 2*1 - 3)/2) + 1 = floor((input-1)/2)+1.
        # This matches ceil(input/2) for odd inputs, and input/2 for even inputs.
        # So padding=1 is a good approximation for 'same' for kernel_size=3.
        if padding == 'same':
            if kernel_size == 3:
                padding_value = 1
            else:
                raise NotImplementedError(f"Only kernel_size=3 'same' padding implemented for now, got {kernel_size}.")
        else:
            padding_value = padding

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding_value, groups=in_channels,
                                   bias=use_bias)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# PyTorch model definition replicating the Keras SimpleXception architecture
class SimpleXception(nn.Module):
    def __init__(self, input_shape=(180, 180, 3), num_classes=2):
        super().__init__()
        # Keras input_shape is (H, W, C), PyTorch is (C, H, W)
        assert input_shape[-1] == 3, "Input shape must be (H, W, C) where C=3"
        self.input_channels = input_shape[-1]
        self.image_height, self.image_width = input_shape[0], input_shape[1]

        # In Keras, `layers.Rescaling(1.0 / 255)` was part of the model.
        # This is handled by `transforms.ToTensor()` in the dataloader, which scales to [0, 1].
        # So, no explicit `Rescaling` layer is needed here.

        # Entry block
        # x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        self.entry_conv = nn.Conv2d(self.input_channels, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.entry_bn = nn.BatchNorm2d(128)
        self.entry_relu = nn.ReLU()

        self.blocks = nn.ModuleList()
        in_channels_block = 128 # After entry block

        for size in [256, 512, 728]:
            block = nn.ModuleList([
                nn.ReLU(),
                KerasLikeSeparableConv2D(in_channels_block, size, kernel_size=3, padding='same', use_bias=False),
                nn.BatchNorm2d(size),
                nn.ReLU(),
                KerasLikeSeparableConv2D(size, size, kernel_size=3, padding='same', use_bias=False),
                nn.BatchNorm2d(size),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ])
            self.blocks.append(nn.Sequential(*block)) # Wrap block layers in Sequential

            # Residual projection (1x1 conv with stride 2 to match MaxPool output size)
            self.blocks.append(
                nn.Conv2d(in_channels_block, size, kernel_size=1, stride=2, padding=0, bias=False)
            )
            in_channels_block = size # Update channels for next block

        # Final block
        self.final_sepconv = KerasLikeSeparableConv2D(in_channels_block, 1024, kernel_size=3, padding='same', use_bias=False)
        self.final_bn = nn.BatchNorm2d(1024)
        self.final_relu = nn.ReLU()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # GlobalAveragePooling2D()
        self.dropout = nn.Dropout(0.25)

        # Output layer
        if num_classes == 2:
            units = 1 # Binary classification, single output for BCEWithLogitsLoss
        else:
            units = num_classes # Multi-class, output logits for CrossEntropyLoss
        self.classifier = nn.Linear(1024, units)

    def forward(self, x):
        # Input tensor is (N, C, H, W). Assumed scaled to [0,1] by dataloader.

        # Entry block
        x = self.entry_conv(x)
        x = self.entry_bn(x)
        x = self.entry_relu(x)

        previous_block_activation = x # Set aside residual

        # Middle blocks
        block_idx = 0
        for i, size in enumerate([256, 512, 728]):
            main_layers = self.blocks[block_idx]
            residual_proj_layer = self.blocks[block_idx + 1]

            x = main_layers(x)
            residual = residual_proj_layer(previous_block_activation)
            x = x + residual # Add back residual
            previous_block_activation = x # Set aside next residual
            block_idx += 2 # Move to next block's main layers and its residual projection

        # Final block
        x = self.final_sepconv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1) # Flatten before dense layer
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs

# Synthetic dataset for fallback in build_dataloader
class SyntheticCatsDogsDataset(Dataset):
    def __init__(self, num_samples=100, image_size=(180, 180), num_channels=3):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image (C, H, W) in [0, 1] range
        image = torch.rand(self.num_channels, self.image_size[0], self.image_size[1])
        # Generate random label (0 or 1 for binary classification)
        label = torch.randint(0, 2, (1,)).item()
        return image, label

# Wrapper dataset to apply transforms after splitting `ImageFolder`
class TransformWrapperDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # ImageFolder returns PIL Image, transform takes PIL Image
        # random_split gives `(data, target)` pair directly from the underlying dataset.
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)

# FL Client Module Functions
def build_model(config: dict) -> torch.nn.Module:
    input_shape = config.get("model_kwargs", {}).get("input_shape", (180, 180, 3))
    num_classes = config.get("model_kwargs", {}).get("num_classes", 2)
    model = SimpleXception(input_shape=input_shape, num_classes=num_classes)
    return model

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    image_size = config.get("model_kwargs", {}).get("input_shape", (180, 180, 3))[:2] # (H, W)

    # Data Augmentation (for train split only)
    # Keras RandomRotation(0.1) means rotating by `[-0.1 * 360, 0.1 * 360]` degrees.
    # So `[-36, 36]` degrees. `transforms.RandomRotation(degrees)` takes a range.
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(18), # Keras 0.1 factor is approx [-18, 18] degrees for rotation
        transforms.ToTensor(), # Scales to [0,1]
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(), # Scales to [0,1]
    ])

    # Keras `image_dataset_from_directory("PetImages", ...)` implies `PetImages` is relative
    # to the working directory or directly provided path.
    # `data_path` is the root, so `PetImages` folder should be inside it.
    dataset_root_path = os.path.join(data_path, "PetImages")

    # Filter corrupted images as per original Keras script's preprocessing step.
    if os.path.exists(dataset_root_path):
        filter_corrupted_images(dataset_root_path)

    try:
        # Load the full dataset (as PIL images initially)
        # No transforms here, they will be applied to subsets after splitting.
        full_dataset = datasets.ImageFolder(
            root=dataset_root_path,
            transform=None, # Transforms will be applied later
            loader=lambda path: Image.open(path).convert('RGB') # Ensure 3 channels
        )

        # Split into train and validation subsets
        # validation_split=0.2 in Keras means 80% train, 20% val.
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset_subset, val_dataset_subset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(1337)
        )

        # Apply appropriate transforms to the respective subsets
        if split == "train":
            dataset = TransformWrapperDataset(train_dataset_subset, transform=train_transforms)
        elif split == "val":
            dataset = TransformWrapperDataset(val_dataset_subset, transform=val_transforms)
        else:
            raise ValueError(f"Unsupported split: {split}. Must be 'train' or 'val'.")

    except FileNotFoundError as e:
        if allow_synthetic_data:
            print(f"Real dataset not found at {dataset_root_path}, using synthetic data.")
            dataset = SyntheticCatsDogsDataset(
                num_samples=batch_size * 10,
                image_size=image_size,
                num_channels=3
            )
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{dataset_root_path}'. "
                f"Set config['allow_synthetic_data'] to True to use synthetic data."
            ) from e
    except Exception as e:
        # Catch other potential errors during dataset loading (e.g., ImageFolder issues)
        if allow_synthetic_data:
            print(f"Error loading real dataset from '{dataset_root_path}': {e}. Using synthetic data.")
            dataset = SyntheticCatsDogsDataset(
                num_samples=batch_size * 10,
                image_size=image_size,
                num_channels=3
            )
        else:
            raise RuntimeError(f"Error loading real dataset from '{dataset_root_path}': {e}") from e


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"), # Only shuffle training data
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 0, # Use half CPU cores if available, otherwise 0
        pin_memory=torch.cuda.is_available() # Pin memory if GPU is available
    )
    return dataloader

def train_step(model: torch.nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    # Get the device of the model parameters
    device = next(model.parameters()).device

    images, labels = batch
    # Move batch to device
    images = images.to(device)
    # Keras used `BinaryCrossentropy(from_logits=True)`, which expects float labels
    # and a single output for binary. So labels should be float and shaped for BCE.
    labels = labels.to(device).float().unsqueeze(1) # Unsqueeze for (N, 1) target

    # Forward pass
    logits = model(images)
    
    # Loss function (BinaryCrossentropy(from_logits=True) in Keras is BCEWithLogitsLoss in PyTorch)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)

    return loss
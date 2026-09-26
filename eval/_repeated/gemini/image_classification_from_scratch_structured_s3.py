import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import transforms
from PIL import Image

# --- Keras Model to PyTorch nn.Module Conversion ---

class SeparableConv2d(nn.Module):
    """
    Equivalent to Keras layers.SeparableConv2D
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        # Depthwise convolution: in_channels -> in_channels, with groups=in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        # Pointwise convolution: in_channels -> out_channels, with kernel_size=1
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class XceptionLikeNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # input_shape is (H, W, C) from Keras, PyTorch expects (C, H, W) for input tensors.
        # The first conv layer's in_channels will be input_shape[2] (C).
        in_channels = input_shape[2]

        # Keras Rescaling (1.0 / 255) is applied explicitly in forward pass
        self.rescale_factor = 1.0 / 255.0

        # Entry block
        self.entry_block = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.previous_block_channels = 128

        self.blocks = nn.ModuleList()
        for i, size in enumerate([256, 512, 728]):
            block_layers = [
                nn.ReLU(inplace=True),
                SeparableConv2d(self.previous_block_channels, size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(size),

                nn.ReLU(inplace=True),
                SeparableConv2d(size, size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(size),

                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ]
            
            # Project residual
            # Keras Conv2D(size, 1, strides=2, padding="same")
            residual_projection = nn.Conv2d(self.previous_block_channels, size, kernel_size=1, stride=2, padding=0, bias=False)
            
            self.blocks.append(nn.ModuleDict({
                'layers': nn.Sequential(*block_layers),
                'residual_projection': residual_projection
            }))
            self.previous_block_channels = size

        self.final_separable_conv = nn.Sequential(
            SeparableConv2d(self.previous_block_channels, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Output layer
        # Keras uses `units=1` for binary classification with `BinaryCrossentropy(from_logits=True)`
        # PyTorch equivalent: nn.Linear(..., 1) for BCEWithLogitsLoss
        final_units = 1 if num_classes == 2 else num_classes
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(1024, final_units)

    def forward(self, inputs):
        # Apply Keras Rescaling (1.0 / 255)
        x = inputs * self.rescale_factor

        # Entry block
        x = self.entry_block(x)
        previous_block_activation = x

        for block_module_dict in self.blocks:
            residual = block_module_dict['residual_projection'](previous_block_activation)
            x = block_module_dict['layers'](x)
            x = x + residual # Add back residual
            previous_block_activation = x # Set aside next residual

        x = self.final_separable_conv(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1) # Flatten except batch dimension
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs


# --- PyTorch Dataset and Dataloader ---

class CatsVsDogsDataset(Dataset):
    """
    A PyTorch Dataset for the Kaggle Cats vs Dogs dataset.
    Handles loading images and filtering corrupted ones, similar to the Keras example.
    """
    def __init__(self, root_dir, image_size):
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_paths = []
        self.labels = [] # 0 for Cat, 1 for Dog

        self._load_and_filter_images()

        if not self.image_paths:
            raise FileNotFoundError(f"No valid images found in {os.path.join(root_dir, 'PetImages')}")

    def _load_and_filter_images(self):
        base_path = os.path.join(self.root_dir, "PetImages")
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"PetImages directory not found at {base_path}")

        for label_name, label_id in [("Cat", 0), ("Dog", 1)]:
            folder_path = os.path.join(base_path, label_name)
            if not os.path.exists(folder_path):
                print(f"Warning: Directory '{folder_path}' not found. Skipping {label_name} images.")
                continue

            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    with open(fpath, "rb") as fobj:
                        # Check for JFIF header to filter corrupted images
                        is_jfif = b"JFIF" in fobj.peek(10)
                except Exception:
                    # Catch any error during file access/peek to be robust
                    continue # Skip problematic files

                if is_jfif:
                    self.image_paths.append(fpath)
                    self.labels.append(label_id)
                # else: Corrupted images are simply skipped, not deleted by the FL client.

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB") # Ensure 3 channels
        except Exception as e:
            # Fallback for images that PIL can't open even if JFIF check passed
            print(f"Error loading image {img_path}: {e}. Returning a black dummy image and label 0.")
            image = Image.new("RGB", self.image_size, color = 'black') # Return a black image
            label = 0 # Default to label 0, or handle as needed by FL runtime
        
        # Transforms will be applied by a wrapper Dataset later
        return image, torch.tensor(label, dtype=torch.long)

class TransformedSubset(Dataset):
    """
    A wrapper Dataset to apply transforms to subsets obtained from random_split.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        # subset[idx] returns (image, label) where image is PIL.Image and label is torch.Tensor
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# --- FL Client Module Functions ---

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    image_size = model_kwargs.get("image_size", (180, 180))
    num_classes = model_kwargs.get("num_classes", 2)
    
    # Keras input_shape is (H, W, C), PyTorch model init needs C from this.
    # DataLoader will output (N, C, H, W) Tensors.
    input_shape = (*image_size, 3) # Assuming RGB images

    model = XceptionLikeNet(input_shape=input_shape, num_classes=num_classes)
    return model

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    image_size = config.get("model_kwargs", {}).get("image_size", (180, 180))
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    val_split_ratio = config.get("dataloader_kwargs", {}).get("validation_split", 0.2)
    seed = config.get("dataloader_kwargs", {}).get("seed", 1337)

    # Keras data augmentation layers to PyTorch transforms:
    # `layers.RandomFlip("horizontal")` -> `transforms.RandomHorizontalFlip(p=0.5)`
    # `layers.RandomRotation(0.1)` -> `transforms.RandomRotation(degrees=(-10, 10))` (approx. 0.1 rad = ~5.7 deg)

    # Note: Keras model had `Rescaling(1./255)` as the first layer.
    # Our PyTorch model `XceptionLikeNet` has `self.rescale_factor = 1.0 / 255.0` applied in forward.
    # Therefore, the data loaded from the DataLoader should be in the [0, 255] range.
    # `transforms.ToTensor()` converts PIL Image (H,W,C) in [0,255] to Tensor (C,H,W) in [0,1].
    # We then multiply by 255.0 to bring it back to [0, 255] range before feeding to the model.

    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5), # Keras RandomFlip "horizontal"
        transforms.RandomRotation(degrees=(-10, 10)), # Keras RandomRotation 0.1 radians
        transforms.ToTensor(), # Scales to [0, 1]
        transforms.Lambda(lambda x: x * 255.0) # Scale to [0, 255] for model's internal Rescaling
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(), # Scales to [0, 1]
        transforms.Lambda(lambda x: x * 255.0) # Scale to [0, 255]
    ])

    # Try to load real data
    full_dataset = None
    try:
        full_dataset = CatsVsDogsDataset(root_dir=data_path, image_size=image_size)
    except FileNotFoundError as e:
        if allow_synthetic_data:
            print(f"Warning: Real data not found ({e}). Generating synthetic data.")
            num_samples = 1000 # Number of synthetic samples for fallback
            # Synthetic images: (batch, C, H, W) in range [0, 255]
            synthetic_images = torch.randint(0, 256, (num_samples, 3, image_size[0], image_size[1]), dtype=torch.float32)
            synthetic_labels = torch.randint(0, 2, (num_samples,), dtype=torch.long) # Binary classification
            full_dataset = TensorDataset(synthetic_images, synthetic_labels)
        else:
            raise FileNotFoundError(
                f"Real data not found at '{os.path.abspath(data_path)}' and 'allow_synthetic_data' is False."
            ) from e
    
    # Check if dataset is empty after loading/filtering
    if len(full_dataset) == 0:
        if allow_synthetic_data:
            print("Warning: Real data dataset is empty. Generating synthetic data.")
            num_samples = 1000
            synthetic_images = torch.randint(0, 256, (num_samples, 3, image_size[0], image_size[1]), dtype=torch.float32)
            synthetic_labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)
            full_dataset = TensorDataset(synthetic_images, synthetic_labels)
        else:
            raise ValueError(
                f"No images found or valid in '{os.path.abspath(data_path)}' and 'allow_synthetic_data' is False."
            )

    # Split dataset into train and validation
    num_total_samples = len(full_dataset)
    num_val_samples = int(val_split_ratio * num_total_samples)
    num_train_samples = num_total_samples - num_val_samples

    if num_train_samples <= 0 or num_val_samples <= 0:
        if num_total_samples > 0:
            print(f"Warning: Dataset size ({num_total_samples} samples) is too small to create both train and val splits with ratio {val_split_ratio}. All samples assigned to {split}.")
            # If dataset is too small, assign all samples to the requested split.
            if split == "train":
                train_dataset = full_dataset
                val_dataset = TensorDataset(torch.empty(0,3,*image_size), torch.empty(0, dtype=torch.long)) # Empty val
            else: # split == "val"
                train_dataset = TensorDataset(torch.empty(0,3,*image_size), torch.empty(0, dtype=torch.long)) # Empty train
                val_dataset = full_dataset
        else:
            raise ValueError("No samples available to create dataset splits.")
    else:
        # Use a generator for reproducibility with seed
        g = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [num_train_samples, num_val_samples], generator=g
        )

    # Apply transforms using the TransformedSubset wrapper
    if split == "train":
        dataset_with_transforms = TransformedSubset(train_dataset, transform=train_transforms)
    elif split == "val":
        dataset_with_transforms = TransformedSubset(val_dataset, transform=val_transforms)
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    dataloader = DataLoader(
        dataset_with_transforms,
        batch_size=batch_size,
        shuffle=(split == "train"), # Shuffle only training data
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 0, # Simple heuristic
        pin_memory=True
    )
    return dataloader


def train_step(model: nn.Module, batch: tuple, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step().
    """
    device = next(model.parameters()).device # Get model's device
    images, labels = batch[0].to(device), batch[1].to(device)

    # Keras used `BinaryCrossentropy(from_logits=True)` for `num_classes=2`.
    # PyTorch equivalent is `nn.BCEWithLogitsLoss`.
    criterion = nn.BCEWithLogitsLoss()

    # Forward pass
    logits = model(images)

    # For binary classification with a single output logit, labels should be float
    # and logits should be squeezed to (batch_size,) if they are (batch_size, 1)
    loss = criterion(logits.squeeze(1), labels.float())

    return loss
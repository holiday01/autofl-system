import os
import warnings

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# List of available models from torchvision.models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class DatasetWithTransform(torch.utils.data.Dataset):
    """
    A wrapper dataset to apply a transform to an existing dataset (e.g., a Subset).
    This is necessary because `torch.utils.data.Subset` does not natively support
    applying different transforms to its underlying dataset based on the split.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve the item from the underlying dataset (e.g., Subset)
        # ImageFolder typically returns a PIL Image and a label.
        # FakeData (with transform=None) automatically applies ToTensor()
        # and returns a Tensor and a label.
        img, label = self.dataset[idx]

        # Apply the specific transform for this dataset/split
        if self.transform:
            img = self.transform(img)
        return img, label

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiates and returns the model.

    Args:
        config: A dictionary containing model configuration, typically
                config.get("model_kwargs", {}) for constructor arguments.

    Returns:
        A torch.nn.Module instance.
    """
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18")
    pretrained = model_kwargs.get("pretrained", False)

    if arch not in model_names:
        raise ValueError(f"Model architecture '{arch}' not supported. Choose from: {', '.join(model_names)}")

    if pretrained:
        print(f"FL client: Using pre-trained model '{arch}'")
        model = models.__dict__[arch](pretrained=True)
    else:
        print(f"FL client: Creating model '{arch}'")
        model = models.__dict__[arch]()

    return model

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Returns a DataLoader for the requested split ("train" or "val").

    Args:
        config: A dictionary containing data configuration.
                - config.get("local", {}).get("batch_size", 16)
                - config.get("local", {}).get("num_workers", 4)
                - config.get("data_path", ".")
                - config.get("val_ratio", 0.2)
                - config.get("allow_synthetic_data", False)
        split: The data split to load ("train" or "val").

    Returns:
        A torch.utils.data.DataLoader instance.

    Raises:
        FileNotFoundError: If real data is unavailable and synthetic data is not allowed.
        ValueError: If an invalid split name is provided.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    num_workers = config.get("local", {}).get("num_workers", 4)
    data_path = config.get("data_path", ".")
    val_ratio = config.get("val_ratio", 0.2)
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"Batch size must be a positive integer, got {batch_size}")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"Number of workers must be a non-negative integer, got {num_workers}")
    if not (0 <= val_ratio <= 1):
        raise ValueError(f"Validation ratio must be between 0 and 1, got {val_ratio}")

    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Transforms for real image data (expecting PIL Image as input)
    train_transform_real = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform_real = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Transform for synthetic data. FakeData's default behavior (when transform=None)
    # is to apply ToTensor() internally. So, we only need to apply normalization afterwards.
    synthetic_data_transform = transforms.Compose([
        normalize,
    ])

    full_dataset = None
    is_synthetic_data = False

    try:
        # Attempt to load ImageFolder from the 'train' subdirectory first,
        # otherwise, assume data_path itself is the root for the unified dataset.
        base_data_dir = os.path.join(data_path, 'train')
        if not os.path.isdir(base_data_dir):
            base_data_dir = data_path

        # Load ImageFolder without initial transforms, transforms will be applied by DatasetWithTransform
        full_dataset = datasets.ImageFolder(base_data_dir)

        if len(full_dataset.classes) == 0:
            raise FileNotFoundError(f"No classes found in dataset at {base_data_dir}")

    except Exception as e:
        if allow_synthetic_data:
            warnings.warn(f"Could not load real data from {data_path} (or its 'train' subdirectory) due to: {e}. "
                          "Using synthetic data. This is typically for testing purposes only.")
            # Approximately ImageNet train + val samples, and 1000 classes.
            num_synthetic_samples = 1331167
            num_classes = 1000
            # FakeData with transform=None applies ToTensor() by default.
            full_dataset = datasets.FakeData(num_synthetic_samples, (3, 224, 224), num_classes, transform=None)
            is_synthetic_data = True
        else:
            raise FileNotFoundError(
                f"Cannot find dataset at {data_path} (or its 'train' subdirectory) and 'allow_synthetic_data' is False. "
                "Set 'allow_synthetic_data: True' in your config to use synthetic data for testing."
            ) from e

    # Split the full dataset into train and validation subsets
    total_size = len(full_dataset)
    if total_size == 0:
        raise ValueError(f"Dataset at {data_path} is empty. Cannot create data loaders.")

    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size

    # Handle cases where one split might become zero for very small datasets
    if train_size == 0 and val_size > 0:
        warnings.warn("Train split size is 0. All data assigned to validation set.")
        val_size = total_size # Assign all to val if train_size became 0
    elif val_size == 0 and train_size > 0:
        warnings.warn("Validation split size is 0. All data assigned to training set.")
        train_size = total_size # Assign all to train if val_size became 0
    elif train_size == 0 and val_size == 0 and total_size > 0:
        # If both are zero but total_size is positive, something is wrong with ratio/sizes
        warnings.warn("Train and validation split sizes are both 0. Reassigning all data to train split.")
        train_size = total_size
        val_size = 0

    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Apply appropriate transforms to the subsets
    if split == "train":
        current_transform = synthetic_data_transform if is_synthetic_data else train_transform_real
        dataset = DatasetWithTransform(train_subset, transform=current_transform)
    elif split == "val":
        current_transform = synthetic_data_transform if is_synthetic_data else val_transform_real
        dataset = DatasetWithTransform(val_subset, transform=current_transform)
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'val'.")

    # Use pin_memory=True if CUDA is available for faster data transfer
    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),  # Shuffle only training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"), # Drop last batch for training to ensure consistent batch sizes
    )

def train_step(model: torch.nn.Module, batch: tuple, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Runs one forward pass, computes the loss, and returns the loss tensor with gradients attached.
    This function should NOT call loss.backward() or optimizer.step().

    Args:
        model: The PyTorch model to train.
        batch: A tuple containing input images and target labels.
        optimizer: The optimizer (not used for backward/step in this function, but passed for API compliance).
        config: A dictionary containing training configuration (not directly used here for loss,
                but available for advanced scenarios).

    Returns:
        The loss tensor with gradients attached.
    """
    # Determine the device of the model parameters
    device = next(model.parameters()).device

    images, target = batch[0], batch[1]
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Define loss function (criterion)
    # Instantiate or retrieve from config if different loss functions are needed
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device) # Ensure criterion is on the same device as the model

    # Compute output
    output = model(images)
    loss = criterion(output, target)

    # Return the loss tensor. The FL runtime will handle `loss.backward()` and `optimizer.step()`.
    return loss
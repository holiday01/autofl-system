import os
import random
import warnings

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split, Dataset

# Preserve model_names from original script for model selection
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Helper Dataset class to apply transforms after random_split
# (Subsets do not allow changing transforms independently)
class DatasetFromSubset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiates and returns the model.

    Args:
        config (dict): Configuration dictionary.
                       Expected to contain 'model_kwargs' for model arguments
                       and 'arch' for model name.

    Returns:
        torch.nn.Module: The instantiated model.
    """
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18") # Default from original script
    pretrained = model_kwargs.get("pretrained", False)

    if arch not in model_names:
        raise ValueError(f"Model architecture '{arch}' not found in torchvision.models.")

    # The original script passes `pretrained` directly to the model constructor.
    # Any additional kwargs for the model constructor can be passed via 'init_kwargs'.
    if pretrained:
        print(f"=> using pre-trained model '{arch}'")
        model = models.__dict__[arch](pretrained=True, **model_kwargs.get("init_kwargs", {}))
    else:
        print(f"=> creating model '{arch}'")
        model = models.__dict__[arch](**model_kwargs.get("init_kwargs", {}))
    
    # The original script applies DataParallel/DistributedDataParallel wrappers.
    # In an FL context, these are typically handled by the FL runtime framework,
    # so we return the base model here.
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Returns a DataLoader for the requested split ("train" or "val").

    Args:
        config (dict): Configuration dictionary.
                       Expected to contain 'local.batch_size', 'data_path',
                       'allow_synthetic_data', 'local.num_workers'.
        split (str): The data split to load ("train" or "val").

    Returns:
        DataLoader: The instantiated DataLoader.

    Raises:
        ValueError: If an invalid split is requested.
        FileNotFoundError: If real data is not found and synthetic data is not allowed.
    """
    if split not in ["train", "val"]:
        raise ValueError(f"Invalid split requested: {split}. Must be 'train' or 'val'.")

    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    num_workers = config.get("local", {}).get("num_workers", 4) # Default from original script

    # Define transforms as in the original script
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    full_dataset = None
    real_data_loaded = False

    # Attempt to load real data
    if os.path.isdir(data_path):
        try:
            # Assuming data_path itself is the root for ImageFolder (e.g., contains class subdirectories)
            full_dataset = datasets.ImageFolder(data_path, transform=None) # No transform initially
            print(f"Loaded real dataset from {data_path} with {len(full_dataset)} samples.")
            real_data_loaded = True
        except Exception as e:
            warnings.warn(f"Could not load ImageFolder from {data_path}: {e}. Trying synthetic data if allowed.")
            full_dataset = None # Reset to trigger synthetic fallback if allowed
    else:
        warnings.warn(f"Data path '{data_path}' not found or is not a directory. Trying synthetic data if allowed.")

    # Synthetic data fallback if real data loading failed or was not attempted
    if not real_data_loaded:
        if allow_synthetic_data:
            print("=> Using synthetic data!")
            # Use fixed sizes for synthetic data, similar to original ImageNet sizes if not specified
            num_synthetic_samples = config.get("synthetic_data_samples", 10000) 
            num_classes = config.get("synthetic_data_classes", 1000) # ImageNet classes
            full_dataset = datasets.FakeData(
                num_synthetic_samples, (3, 224, 224), num_classes, transforms.ToTensor()
            )
        else:
            raise FileNotFoundError(
                f"Real data not found at '{data_path}' and synthetic data is not allowed. "
                "Set config['allow_synthetic_data'] to True to use synthetic data fallback."
            )

    # Apply random_split to the single full_dataset as required by the rules.
    # The split ratio can be configured, default to 80/20.
    train_split_ratio = config.get("data_split_ratio", 0.8)
    total_size = len(full_dataset)
    train_size = int(total_size * train_split_ratio)
    val_size = total_size - train_size
    
    # Use a fixed seed for random_split to ensure the train/val split is consistent
    # across calls and clients if using the same seed.
    g = torch.Generator().manual_seed(config.get("seed", 42)) 
    train_subset_indices, val_subset_indices = random_split(range(total_size), [train_size, val_size], generator=g)

    # Create Subsets from the full dataset using the generated indices
    train_base_subset = Subset(full_dataset, train_subset_indices.indices)
    val_base_subset = Subset(full_dataset, val_subset_indices.indices)

    # Wrap these subsets with specific transforms using the helper class
    train_data = DatasetFromSubset(train_base_subset, train_transform)
    val_data = DatasetFromSubset(val_base_subset, val_transform)

    if split == "train":
        dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True, # Shuffle for training
            num_workers=num_workers,
            pin_memory=True,
        )
    else: # split == "val"
        dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False, # No shuffle for validation
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloader


def train_step(model: torch.nn.Module, batch: tuple, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Runs one forward pass only and returns the loss tensor WITH grad attached.
    This function does NOT call loss.backward() or optimizer.step().

    Args:
        model (torch.nn.Module): The model to train.
        batch (tuple): A tuple containing (images, targets).
        optimizer (torch.optim.Optimizer): The optimizer (not used for step/zero_grad here,
                                           but could be used for other logic if needed).
        config (dict): Configuration dictionary (not directly used in this simple step,
                       but available for more complex scenarios like custom loss functions).

    Returns:
        torch.Tensor: The computed loss tensor with gradients attached.
    """
    images, target = batch
    
    # Move tensors to the device of the model parameters
    # This assumes the model has already been moved to the correct device by the FL runtime.
    device = next(model.parameters()).device
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Set model to training mode (important for BatchNorm, Dropout, etc.)
    model.train()

    # Compute output
    output = model(images)
    
    # Define loss function (criterion)
    # The original script uses CrossEntropyLoss.
    # Instantiate it here; for efficiency, it could be instantiated once per client in a higher scope.
    criterion = nn.CrossEntropyLoss().to(device)

    # Compute loss
    loss = criterion(output, target)

    return loss
import os
import random
import warnings

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Define model_names similar to the original script for build_model
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    - Uses config.get("model_kwargs", {}) for constructor args.
    """
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18")
    pretrained = model_kwargs.get("pretrained", False)

    if arch not in model_names:
        raise ValueError(f"Model architecture '{arch}' not supported. "
                         f"Available models: {', '.join(model_names)}")

    if pretrained:
        model = models.__dict__[arch](pretrained=True)
    else:
        model = models.__dict__[arch]()

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    - Read batch_size from config.get("local", {}).get("batch_size", 16).
    - Read data_path from config.get("data_path", ".").
    - Use random_split to produce train/val subsets from a single dataset.
      (Note: This implementation adheres to the original script's logic by
      looking for separate 'train' and 'val' directories if real data is used,
      which is common for ImageNet. If such directories don't exist, it falls
      back to synthetic data if allowed.)
    - Include a synthetic data fallback (torch.randn/randint) for the case
      where the real dataset is unavailable, but it MUST be gated on
      config.get("allow_synthetic_data", False).
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    num_workers = config.get("local", {}).get("num_workers", 4)
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset_size = 0
    if split == "train":
        data_dir = os.path.join(data_path, 'train')
        dataset_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        dataset_size = 1281167  # ImageNet train size from original script's FakeData
    elif split == "val":
        data_dir = os.path.join(data_path, 'val')
        dataset_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        dataset_size = 50000  # ImageNet val size from original script's FakeData
    else:
        raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

    dataset = None
    if os.path.exists(data_dir):
        try:
            dataset = datasets.ImageFolder(data_dir, dataset_transforms)
            # ImageFolder typically raises an error if root is empty or invalid,
            # so checking `if not dataset` directly after creation is often redundant.
            # The exception handling below covers loading issues.
        except Exception as e:
            warnings.warn(f"Failed to load ImageFolder from {data_dir}: {e}. "
                          f"Attempting synthetic data fallback for split '{split}'.")
            dataset = None

    if dataset is None:
        if allow_synthetic_data:
            warnings.warn(f"Using synthetic data for split '{split}' as real data is "
                          "unavailable or failed to load, and 'allow_synthetic_data' is True.")
            # Parameters from original script's FakeData for ImageNet-like data
            dataset = datasets.FakeData(dataset_size, (3, 224, 224), 1000, dataset_transforms)
        else:
            raise FileNotFoundError(
                f"Real data not found at {data_dir} for split '{split}' and "
                f"'allow_synthetic_data' is False. "
                "Set 'allow_synthetic_data': True in config to use synthetic data."
            )

    shuffle = (split == "train")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    - Move tensors to the device of the model parameters.
    """
    # Set model to training mode (important for BatchNorm, Dropout)
    model.train()

    images, target = batch

    # Move tensors to the device of the model parameters
    device = next(model.parameters()).device
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Define loss function (criterion)
    # The original script uses nn.CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Compute output
    output = model(images)
    loss = criterion(output, target)

    return loss
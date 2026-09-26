import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader, random_split

# model_names from original script
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Custom Dataset wrapper to apply transforms after random_split
class _DatasetWithTransforms(torch.utils.data.Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)

    @property
    def classes(self):
        # Access classes from the original dataset within the subset
        return self.subset_dataset.dataset.classes

    @property
    def class_to_idx(self):
        # Access class_to_idx from the original dataset within the subset
        return self.subset_dataset.dataset.class_to_idx


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    - Use config.get("model_kwargs", {}) for constructor args.
    """
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18")
    pretrained = model_kwargs.get("pretrained", False)

    if arch not in models.__dict__ or not callable(models.__dict__[arch]):
        raise ValueError(f"Model architecture '{arch}' not found in torchvision.models or not callable. "
                         f"Available models: {', '.join(model_names)}")

    if pretrained:
        print(f"=> using pre-trained model '{arch}'")
        model = models.__dict__[arch](pretrained=True)
    else:
        print(f"=> creating model '{arch}'")
        model = models.__dict__[arch]()

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    - Read batch_size from config.get("local", {}).get("batch_size", 16).
    - Read data_path from config.get("data_path", ".").
    - Use random_split to produce train/val subsets from a single dataset.
    - Include a synthetic data fallback (torch.randn/randint) for the case
      where the real dataset is unavailable, but it MUST be gated on
      config.get("allow_synthetic_data", False).
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    base_data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    num_workers = config.get("local", {}).get("num_workers", 0) # Default to 0 for simplicity in FL context

    # ImageNet normalization from original script
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Define transforms for train and validation splits
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

    final_dataset = None
    shuffle = False

    if allow_synthetic_data:
        print(f"=> Using synthetic data for {split} split.")
        # ImageNet typically has 1000 classes.
        num_classes = config.get("model_kwargs", {}).get("num_classes", 1000)
        
        # For synthetic data, we can directly create a FakeData instance with the correct transform.
        # The 'random_split from a single dataset' rule doesn't strictly apply when the dataset is synthetic
        # and can be generated directly for each split.
        if split == "train":
            synthetic_data_size = config.get("local", {}).get("synthetic_train_size", 60000) 
            final_dataset = datasets.FakeData(synthetic_data_size, (3, 224, 224), num_classes, transform=train_transform)
            shuffle = True
        elif split == "val":
            synthetic_data_size = config.get("local", {}).get("synthetic_val_size", 10000) 
            final_dataset = datasets.FakeData(synthetic_data_size, (3, 224, 224), num_classes, transform=val_transform)
            shuffle = False
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

    else: # Real data path
        # The original script uses `os.path.join(args.data, 'train')` for training data.
        # To satisfy "random_split to produce train/val subsets from a single dataset",
        # we will load the client's local training data (assuming it's in a 'train' subdirectory
        # of `base_data_path`) as the *single dataset* and then split it into local train/val.
        
        dataset_for_split_path = os.path.join(base_data_path, 'train')
        if not os.path.exists(dataset_for_split_path):
            raise FileNotFoundError(f"Real data 'train' split not found at {dataset_for_split_path}. "
                                    f"Set 'allow_synthetic_data: true' in config or provide valid 'data_path'.")
        
        print(f"=> Loading real data from {dataset_for_split_path} for splitting.")
        full_client_dataset = datasets.ImageFolder(dataset_for_split_path)

        if len(full_client_dataset) == 0:
            raise ValueError(f"Dataset at {dataset_for_split_path} is empty. Cannot create splits.")

        # Use random_split to create train/val subsets from this single dataset
        train_ratio = config.get("local", {}).get("train_split_ratio", 0.8) 
        train_size = int(train_ratio * len(full_client_dataset))
        val_size = len(full_client_dataset) - train_size
        
        if train_size == 0 or val_size == 0:
            warnings.warn(f"One of the splits (train/val) has zero size. "
                          f"train_size={train_size}, val_size={val_size}. "
                          f"Consider adjusting 'train_split_ratio' or ensuring sufficient data at {dataset_for_split_path}.")

        # Ensure reproducibility for splitting if a seed is provided in config
        seed = config.get("seed", None)
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
            train_subset, val_subset = random_split(full_client_dataset, [train_size, val_size], generator=g)
        else:
            train_subset, val_subset = random_split(full_client_dataset, [train_size, val_size])

        if split == "train":
            final_dataset = _DatasetWithTransforms(train_subset, train_transform)
            shuffle = True
        elif split == "val":
            final_dataset = _DatasetWithTransforms(val_subset, val_transform)
            shuffle = False
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

    if final_dataset is None:
        raise RuntimeError("Final dataset could not be created. This should not happen.")

    return DataLoader(final_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    - Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    - Move tensors to the device of the model parameters.
    """
    images, target = batch

    # Get the device from the model's parameters (assuming model is already on the correct device)
    device = next(model.parameters()).device

    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Define loss function (criterion). Original script uses CrossEntropyLoss.
    criterion = nn.CrossEntropyLoss()

    # Compute output
    output = model(images)
    loss = criterion(output, target)

    return loss
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
from torch.utils.data import DataLoader, Subset

# Define model_names globally as in the original script
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18")
    pretrained = model_kwargs.get("pretrained", False)

    if arch not in model_names:
        raise ValueError(f"Model architecture '{arch}' not supported. Choose from: {', '.join(model_names)}")

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
    """
    if split not in ["train", "val"]:
        raise ValueError(f"Split must be 'train' or 'val', but got '{split}'")

    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    
    # Use a specific seed for reproducibility of splits and dataloader shuffling
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    # Generator for DataLoader shuffle
    g = torch.Generator()
    g.manual_seed(seed)

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

    # Rule: "Use random_split to produce train/val subsets from a single dataset."
    # We assume `data_path` points to the root directory of the client's data,
    # which contains subdirectories for classes (e.g., data_path/classA/, data_path/classB/).
    # This implies the client performs its own local train/val split from its total available data.
    
    train_dataset = None
    val_dataset = None

    try:
        if allow_synthetic_data:
            print("=> Using synthetic data for local train/val split.")
            num_synthetic_samples = config.get("synthetic_samples", 10000) 
            num_classes = config.get("num_classes", 1000) 
            
            # Since FakeData doesn't have an underlying common dataset to split,
            # we directly create two FakeData instances with appropriate sizes.
            train_ratio = config.get("local", {}).get("train_val_split_ratio", 0.8)
            train_size = int(train_ratio * num_synthetic_samples)
            val_size = num_synthetic_samples - train_size

            train_dataset = datasets.FakeData(
                train_size, (3, 224, 224), num_classes, train_transform
            )
            val_dataset = datasets.FakeData(
                val_size, (3, 224, 224), num_classes, val_transform
            )
            print(f"Synthetic data: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}")
        else:
            print(f"=> Loading real dataset from {data_path} for local train/val split.")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data path not found: {data_path}. "
                                        "Set 'allow_synthetic_data' to True in config for synthetic fallback.")
            
            # Create a base ImageFolder without transforms to get total length for splitting indices
            full_dataset_base = datasets.ImageFolder(root=data_path, transform=None)
            if len(full_dataset_base) == 0:
                 raise ValueError(f"No samples found in {data_path}. Please check data path and structure.")

            # Define split sizes
            train_ratio = config.get("local", {}).get("train_val_split_ratio", 0.8)
            train_size = int(train_ratio * len(full_dataset_base))
            val_size = len(full_dataset_base) - train_size
            
            # Use torch.utils.data.random_split to get indices for train and val splits
            # We split the range of indices, then use these indices with datasets that have specific transforms.
            train_indices, val_indices = torch.utils.data.random_split(
                list(range(len(full_dataset_base))), 
                [train_size, val_size],
                generator=g # Ensure reproducibility of the split
            )

            # Create two ImageFolder instances, each with its specific transform.
            # Then use Subset to select the correct indices.
            full_dataset_train_transforms = datasets.ImageFolder(root=data_path, transform=train_transform)
            full_dataset_val_transforms = datasets.ImageFolder(root=data_path, transform=val_transform)

            train_dataset = Subset(full_dataset_train_transforms, train_indices)
            val_dataset = Subset(full_dataset_val_transforms, val_indices)
            print(f"Loaded real data: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}")

    except FileNotFoundError as e:
        if not allow_synthetic_data:
            raise e # Re-raise if synthetic data is not allowed
        else:
            warnings.warn(f"Real data loading failed: {e}. Falling back to synthetic data as allowed.")
            # Re-attempt synthetic data generation if there was an issue in initial real data loading
            num_synthetic_samples = config.get("synthetic_samples", 10000)
            num_classes = config.get("num_classes", 1000)
            train_ratio = config.get("local", {}).get("train_val_split_ratio", 0.8)
            train_size = int(train_ratio * num_synthetic_samples)
            val_size = num_synthetic_samples - train_size
            train_dataset = datasets.FakeData(train_size, (3, 224, 224), num_classes, train_transform)
            val_dataset = datasets.FakeData(val_size, (3, 224, 224), num_classes, val_transform)
            print(f"Successfully generated synthetic data after fallback: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}")
    except Exception as e:
        # Catch other potential errors during real data loading, like ImageFolder structure issues
        if not allow_synthetic_data:
            raise e
        else:
            warnings.warn(f"Real data loading failed for an unexpected reason: {e}. Falling back to synthetic data as allowed.")
            # Re-attempt synthetic data generation if there was an issue in initial real data loading
            num_synthetic_samples = config.get("synthetic_samples", 10000)
            num_classes = config.get("num_classes", 1000)
            train_ratio = config.get("local", {}).get("train_val_split_ratio", 0.8)
            train_size = int(train_ratio * num_synthetic_samples)
            val_size = num_synthetic_samples - train_size
            train_dataset = datasets.FakeData(train_size, (3, 224, 224), num_classes, train_transform)
            val_dataset = datasets.FakeData(val_size, (3, 224, 224), num_classes, val_transform)
            print(f"Successfully generated synthetic data after fallback: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}")

    dataset = train_dataset if split == "train" else val_dataset

    # Workers from config, default 4 as in original
    num_workers = config.get("local", {}).get("num_workers", 4)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"), # Shuffle only for training split
        num_workers=num_workers,
        pin_memory=True,
        generator=g if split == "train" else None # Pass generator only for train shuffle
    )

    return dataloader


def train_step(model: torch.nn.Module, batch, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    """
    images, target = batch

    # Move tensors to the device of the model parameters
    # Assuming all model parameters are on the same device
    device = next(model.parameters()).device
    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Set model to train mode
    model.train()

    # Compute output
    output = model(images)
    
    # Define criterion (loss function)
    # The original script uses nn.CrossEntropyLoss for ImageNet classification.
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device) # Ensure criterion is on the same device as model/data

    # Compute loss
    loss = criterion(output, target)

    # The FL runtime will handle loss.backward() and optimizer.step()
    return loss

```
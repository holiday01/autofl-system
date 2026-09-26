"""
Auto-generated FL client module.
Original script: PyTorch ImageNet Training (from pytorch/examples)

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Helper list from original script to identify available models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the model based on the provided configuration.
    """
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18")
    pretrained = model_kwargs.get("pretrained", False)

    if arch not in model_names:
        raise ValueError(f"Model architecture '{arch}' not found in torchvision.models.")

    if pretrained:
        print(f"=> using pre-trained model '{arch}'")
        model = models.__dict__[arch](pretrained=True)
    else:
        print(f"=> creating model '{arch}'")
        model = models.__dict__[arch]()
    
    # The original script assumes ImageNet's 1000 classes.
    # If a different number of classes is needed, the final layer of the model
    # would need to be modified here, e.g., model.fc = nn.Linear(model.fc.in_features, num_classes).
    # For this conversion, we stick to the original script's implicit assumption of 1000 classes.

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a DataLoader for the specified split (train/val).
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 256))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory = local.get("pin_memory", True) # Original script uses True

    data_path = config.get("data_path", "imagenet")
    dummy_data = config.get("dummy_data", False) # Use 'dummy_data' in config to enable FakeData

    if dummy_data:
        print("=> Dummy data is used!")
        if split == "train":
            dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
            shuffle = True # FakeData doesn't have a concept of order, but for consistency
        else: # val
            dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
            shuffle = False
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if split == "train":
            traindir = os.path.join(data_path, 'train')
            dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            shuffle = True
        elif split == "val": # Use "val" for validation split
            valdir = os.path.join(data_path, 'val')
            dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            shuffle = False
        else:
            raise ValueError(f"Unsupported split: {split}. Expected 'train' or 'val'.")
    
    # For a single FL client, we typically don't use DistributedSampler.
    # The FL runtime manages distribution across clients.
    sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        sampler=sampler,
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer, # Kept for signature consistency, but not used directly as per contract
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    
    # The original script's train function expects (images, target) tuple.
    # We'll assume the DataLoader provides a tuple (images, target).
    if isinstance(batch, (list, tuple)):
        images, target = batch
    else:
        # This branch is unlikely to be hit with ImageFolder/FakeData,
        # but included for robustness if batch format changes.
        images = batch.get("input", batch.get("x", batch.get("image")))
        target = batch.get("label", batch.get("y", batch.get("target")))

    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Compute output
    output = model(images)
    
    # Define loss function (criterion)
    # Following the example, instantiate criterion inside train_step.
    criterion = nn.CrossEntropyLoss()
    
    loss = criterion(output, target)
    return loss
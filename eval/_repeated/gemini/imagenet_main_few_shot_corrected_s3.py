"""
Auto-generated FL client module.
Original script: pytorch/examples/imagenet/main.py (a common reference implementation)

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


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds the model based on the provided configuration.
    """
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18")  # Default from original script
    pretrained = model_kwargs.get("pretrained", False)  # Default from original script

    if pretrained:
        print(f"=> using pre-trained model '{arch}'")
        model = models.__dict__[arch](pretrained=True)
    else:
        print(f"=> creating model '{arch}'")
        model = models.__dict__[arch]()

    # In an FL client module, the FL runtime is responsible for placing the model
    # on the correct device(s) and potentially wrapping it for parallelism.
    # So, we simply return the raw model.
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds a DataLoader for the specified split (train or val).
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 256))  # Default from original script
    num_workers = local.get("num_workers", config.get("num_workers", 4))  # Default from original script
    pin_memory = local.get("pin_memory", True)

    data_path = config.get("data_path", "imagenet")  # Default from original script
    dummy_data = config.get("dummy_data", False)  # Added config flag for dummy data

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = None
    if dummy_data:
        print("=> Dummy data is used!")
        if split == "train":
            dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        elif split == "val":
            dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
        else:
            raise ValueError(f"Unknown split: {split}")
    else:
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
        elif split == "val":
            valdir = os.path.join(data_path, 'val')
            dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            raise ValueError(f"Unknown split: {split}")

    # For a standalone FL client module, we don't assume distributed training *within* the client.
    # The FL framework itself orchestrates distribution across clients.
    # So, we use shuffle=True for train, False for val, and no explicit sampler.
    shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),  # Only pin memory if CUDA is available
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    # Ensure model is in training mode (the FL runtime might also do this)
    model.train()

    # Get the device where the model currently resides
    device = next(model.parameters()).device

    # Move batch data to the model's device
    # Assuming batch is (images, target) based on typical ImageNet DataLoader output
    if isinstance(batch, (list, tuple)):
        images, targets = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        # Fallback for dict-like batches, common in some ML pipelines
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        images = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # Compute output
    outputs = model(images)

    # Define the loss criterion (CrossEntropyLoss from original script)
    # Instantiate it here for simplicity and to follow the example's pattern.
    criterion = nn.CrossEntropyLoss()
    # Move criterion to device if it holds state that needs device placement (CrossEntropyLoss typically doesn't,
    # but it's a good general practice for modules that might)
    criterion = criterion.to(device)

    # Compute loss
    loss = criterion(outputs, targets)

    return loss
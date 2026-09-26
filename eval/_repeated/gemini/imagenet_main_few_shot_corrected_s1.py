"""
Auto-generated FL client module.
Original script: pytorch/examples/imagenet/main.py

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


# --- Helper for build_model: list of available models ---
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the PyTorch model based on the provided configuration.
    The FL runtime will handle moving the model to the appropriate device
    and potentially wrapping it for distributed training.
    """
    model_kwargs = config.get("model_kwargs", {})
    arch = model_kwargs.get("arch", "resnet18")
    pretrained = model_kwargs.get("pretrained", False)

    if arch not in model_names:
        raise ValueError(f"Model architecture '{arch}' not supported. Choose from: {model_names}")

    print(f"=> {'Using pre-trained' if pretrained else 'Creating'} model '{arch}'")
    model = models.__dict__[arch](pretrained=pretrained)

    # The original script handles DataParallel/DistributedDataParallel.
    # In an FL client module, the FL runtime is responsible for model distribution
    # and device placement. We return the base model.
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a PyTorch DataLoader for the specified data split
    (e.g., "train", "val").
    """
    local_config = config.get("local", {})
    batch_size = local_config.get("batch_size", config.get("batch_size", 256))
    num_workers = local_config.get("num_workers", config.get("num_workers", 4))
    pin_memory = local_config.get("pin_memory", True)
    data_path = config.get("data_path", "imagenet") # Default from original script

    dataset_kwargs = config.get("dataset_kwargs", {})
    dummy_data = dataset_kwargs.get("dummy", False) # Reflects `args.dummy`

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if dummy_data:
        print("=> Dummy data is used!")
        if split == "train":
            dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        else: # "val"
            dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        if split == "train":
            dataset_path = os.path.join(data_path, 'train')
            dataset = datasets.ImageFolder(
                dataset_path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        elif split == "val":
            dataset_path = os.path.join(data_path, 'val')
            dataset = datasets.ImageFolder(
                dataset_path,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            raise ValueError(f"Unsupported split: {split}. Choose 'train' or 'val'.")

    # Samplers for distributed training (like DistributedSampler) are typically
    # managed by the FL runtime if distributed training is enabled.
    # We set shuffle based on the split type for standard DataLoader behavior.
    shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    model.train() # Ensure model is in training mode

    # Determine the device of the model parameters to move batch data there
    device = next(model.parameters()).device

    # Move batch data (images, target) to the correct device
    if isinstance(batch, (list, tuple)):
        images, target = batch[0], batch[1]
    elif isinstance(batch, dict):
        images  = batch.get("input", batch.get("x", batch.get("image")))
        target = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    images = images.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # Compute output
    output = model(images)

    # Define and compute loss
    # The original script uses nn.CrossEntropyLoss
    criterion = nn.CrossEntropyLoss().to(device)
    loss = criterion(output, target)

    return loss
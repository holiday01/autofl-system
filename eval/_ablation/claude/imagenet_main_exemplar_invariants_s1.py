"""
Auto-generated FL client module.
Original script: PyTorch ImageNet Training (torchvision example)

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
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
from torch.utils.data import DataLoader, TensorDataset


def build_model(config: dict) -> nn.Module:
    arch = config.get("arch", "resnet18")
    pretrained = config.get("pretrained", False)
    num_classes = config.get("num_classes", 1000)

    weights = "DEFAULT" if pretrained else None
    model = models.__dict__[arch](weights=weights)

    if num_classes != 1000:
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                last = model.classifier[-1]
                model.classifier[-1] = nn.Linear(last.in_features, num_classes)
            elif isinstance(model.classifier, nn.Linear):
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  32))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory",  True)
    num_classes = config.get("num_classes", 1000)

    data_path       = config.get("data_path", "imagenet")
    allow_synthetic = config.get("allow_synthetic_data", False)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        subdir = os.path.join(data_path, "train")
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        subdir = os.path.join(data_path, "val")

    if os.path.isdir(subdir):
        dataset = datasets.ImageFolder(subdir, transform=transform)
    else:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"ImageNet {split} directory not found at '{subdir}'. "
                "Set config['allow_synthetic_data'] = True to use a synthetic fallback."
            )
        n = 256 if split == "train" else 64
        X = torch.randn(n, 3, 224, 224)
        y = torch.randint(0, num_classes, (n,))
        dataset = TensorDataset(X, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs  = batch.get("input",  batch.get("x",      batch.get("image")))
        targets = batch.get("label",  batch.get("y",      batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    return loss
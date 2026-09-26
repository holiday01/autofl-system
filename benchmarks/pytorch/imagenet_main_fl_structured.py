"""
FL client for imagenet_main.py (PyTorch ImageNet training).
Structured conversion: ResNet18 with synthetic FakeData fallback.
"""
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split


def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    arch = kwargs.get("arch", "resnet18")
    return models.__dict__[arch]()


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", "imagenet")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    try:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224) if split == "train" else transforms.Resize(256),
            transforms.RandomHorizontalFlip() if split == "train" else transforms.CenterCrop(224),
            transforms.ToTensor(), normalize,
        ])
        split_dir = os.path.join(data_path, "train" if split == "train" else "val")
        dataset = datasets.ImageFolder(split_dir, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=2, pin_memory=False)
    except Exception:
        n = 200
        data = torch.randn(n, 3, 224, 224)
        targets = torch.randint(0, 1000, (n,))
        full_ds = TensorDataset(data, targets)
        n_val = max(1, int(0.1 * n))
        train_ds, val_ds = random_split(
            full_ds, [n - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_ds if split == "train" else val_ds
        return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """One forward pass — returns CrossEntropyLoss with grad_fn attached.
    The FL runtime is responsible for loss.backward() and optimizer.step().
    """
    device = next(model.parameters()).device
    images, targets = batch[0].to(device), batch[1].to(device)
    outputs = model(images)
    return nn.CrossEntropyLoss()(outputs, targets)

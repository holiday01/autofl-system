import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv1 = SeparableConv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.sepconv2 = SeparableConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(x)
        x = self.bn1(self.sepconv1(x))
        x = F.relu(x)
        x = self.bn2(self.sepconv2(x))
        x = self.pool(x)
        return x + residual


class MiniXception(nn.Module):
    def __init__(self, num_classes=2, dropout=0.25):
        super().__init__()
        self.entry_conv = nn.Conv2d(3, 128, 3, stride=2, padding=1, bias=False)
        self.entry_bn = nn.BatchNorm2d(128)

        self.block1 = ResidualBlock(128, 256)
        self.block2 = ResidualBlock(256, 512)
        self.block3 = ResidualBlock(512, 728)

        self.top_conv = SeparableConv2d(728, 1024, 3, padding=1)
        self.top_bn = nn.BatchNorm2d(1024)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        units = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(1024, units)

    def forward(self, x):
        x = F.relu(self.entry_bn(self.entry_conv(x)))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = F.relu(self.top_bn(self.top_conv(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


class _SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        base = subset.dataset
        self.samples = [base.imgs[i] for i in subset.indices]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    pet_images_path = os.path.join(data_path, "PetImages")

    train_transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(36),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
    ])

    if os.path.isdir(pet_images_path):
        base_dataset = datasets.ImageFolder(pet_images_path)
        n = len(base_dataset)
        val_size = int(0.2 * n)
        train_size = n - val_size
        train_sub, val_sub = random_split(
            base_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(1337),
        )
        if split == "val":
            ds = _SubsetWithTransform(val_sub, val_transform)
            return DataLoader(ds, batch_size=batch_size, shuffle=False)
        ds = _SubsetWithTransform(train_sub, train_transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"PetImages dataset not found at '{pet_images_path}'. "
            "Set config['allow_synthetic_data']=True to use synthetic data for testing."
        )
    n_samples = 100
    images = torch.randn(n_samples, 3, 180, 180)
    labels = torch.randint(0, 2, (n_samples,))
    dataset = torch.utils.data.TensorDataset(images, labels)
    val_size = int(0.2 * n_samples)
    train_size = n_samples - val_size
    train_sub, val_sub = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1337),
    )
    chosen = val_sub if split == "val" else train_sub
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    images, labels = batch
    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device).float()
    logits = model(images).squeeze(1)
    return F.binary_cross_entropy_with_logits(logits, labels)
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image as PILImage


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sep1 = SeparableConv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.sep2 = SeparableConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.proj = nn.Conv2d(in_channels, out_channels, 1, stride=2)

    def forward(self, x):
        prev = x
        x = F.relu(x)
        x = self.bn1(self.sep1(x))
        x = F.relu(x)
        x = self.bn2(self.sep2(x))
        x = self.pool(x)
        return x + self.proj(prev)


class MiniXception(nn.Module):
    """PyTorch equivalent of the Keras mini-Xception built by make_model()."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.entry_conv = nn.Conv2d(3, 128, 3, stride=2, padding=1)
        self.entry_bn = nn.BatchNorm2d(128)

        self.block1 = _ResBlock(128, 256)
        self.block2 = _ResBlock(256, 512)
        self.block3 = _ResBlock(512, 728)

        self.top_sep = SeparableConv2d(728, 1024, 3, padding=1)
        self.top_bn = nn.BatchNorm2d(1024)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)
        units = 1 if num_classes == 2 else num_classes
        self.fc = nn.Linear(1024, units)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.entry_bn(self.entry_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.top_bn(self.top_sep(x)))
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class _SubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = PILImage.open(path).convert("RGB")
        return self.transform(img), label


def build_model(config):
    num_classes = config.get("num_classes", 2)
    return MiniXception(num_classes=num_classes)


def build_dataloader(config, split="train"):
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)

    def _to_float_tensor(img):
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()

    aug_transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(36),
        transforms.Lambda(_to_float_tensor),
    ])
    plain_transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.Lambda(_to_float_tensor),
    ])

    base = ImageFolder(root=data_path)
    n_val = int(0.2 * len(base))
    n_train = len(base) - n_val
    train_subset, val_subset = random_split(
        base, [n_train, n_val], generator=torch.Generator().manual_seed(1337)
    )

    if split == "train":
        dataset = _SubsetWrapper(train_subset, aug_transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset = _SubsetWrapper(val_subset, plain_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_step(model, batch, optimizer, config):
    images, labels = batch
    optimizer.zero_grad()
    logits = model(images).squeeze(1)
    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    loss.backward()
    optimizer.step()
    return loss.item()
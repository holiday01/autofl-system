import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


IMAGE_SIZE = (180, 180)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MiniXception(nn.Module):
    def __init__(self, input_shape=(180, 180, 3), num_classes=2):
        super().__init__()
        in_channels = input_shape[2]

        self.entry_conv = nn.Conv2d(in_channels, 128, 3, stride=2, padding=1, bias=False)
        self.entry_bn = nn.BatchNorm2d(128)

        self.res_blocks = nn.ModuleList()
        self.res_projections = nn.ModuleList()

        prev_ch = 128
        for size in [256, 512, 728]:
            self.res_blocks.append(nn.Sequential(
                nn.ReLU(),
                DepthwiseSeparableConv2d(prev_ch, size, 3, padding=1),
                nn.BatchNorm2d(size),
                nn.ReLU(),
                DepthwiseSeparableConv2d(size, size, 3, padding=1),
                nn.BatchNorm2d(size),
                nn.MaxPool2d(3, stride=2, padding=1),
            ))
            self.res_projections.append(
                nn.Conv2d(prev_ch, size, 1, stride=2, padding=0, bias=False)
            )
            prev_ch = size

        self.top_conv = DepthwiseSeparableConv2d(728, 1024, 3, padding=1)
        self.top_bn = nn.BatchNorm2d(1024)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)

        units = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(1024, units)

    def forward(self, x):
        x = x / 255.0
        x = self.entry_conv(x)
        x = self.entry_bn(x)
        x = F.relu(x)

        for block, proj in zip(self.res_blocks, self.res_projections):
            residual = proj(x)
            x = block(x) + residual

        x = self.top_conv(x)
        x = self.top_bn(x)
        x = F.relu(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


class _CatsDogsDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.samples = []
        self.transform = transform
        for label_idx, class_name in enumerate(["Cat", "Dog"]):
            class_dir = os.path.join(data_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                fpath = os.path.join(class_dir, fname)
                self.samples.append((fpath, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
        arr = np.array(img, dtype=np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.float32)


class _TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


def build_model(config):
    num_classes = config.get("num_classes", 2)
    return MiniXception(input_shape=(180, 180, 3), num_classes=num_classes)


def build_dataloader(config, split="train"):
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)

    full_dataset = _CatsDogsDataset(data_path)
    n = len(full_dataset)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    generator = torch.Generator().manual_seed(1337)
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    if split == "train":
        aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(36),
        ])
        dataset = _TransformSubset(train_subset, aug)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return DataLoader(val_subset, batch_size=batch_size, shuffle=False)


def train_step(model, batch, optimizer, config):
    model.train()
    images, labels = batch
    optimizer.zero_grad()
    logits = model(images)
    if logits.shape[1] == 1:
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels)
    else:
        loss = F.cross_entropy(logits, labels.long())
    loss.backward()
    optimizer.step()
    return loss.item()
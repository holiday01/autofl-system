import numpy as np
import keras
from keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

num_classes = 10
input_shape = (28, 28, 1)


class MNISTConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * 5 * 5, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return F.softmax(self.fc(x), dim=1)


def build_model(config):
    nc = config.get("num_classes", num_classes)
    return MNISTConvNet(num_classes=nc)


def build_dataloader(config, split="train"):
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = torchvision.datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform
    )
    n_train = int(0.9 * len(full_dataset))
    n_val = len(full_dataset) - n_train
    train_set, val_set = random_split(full_dataset, [n_train, n_val])
    dataset = train_set if split == "train" else val_set
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config):
    model.train()
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = F.nll_loss(outputs.clamp(min=1e-7).log(), targets)
    loss.backward()
    optimizer.step()
    return loss.item()
import os
import numpy as np
import keras
from keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

num_classes = 10
input_shape = (28, 28, 1)


class MNISTConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1600, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


class _MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def build_model(config):
    return MNISTConvNet()


def build_dataloader(config, split="train"):
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    (x_train, y_train), _ = keras.datasets.mnist.load_data(
        path=os.path.join(data_path, "mnist.npz")
    )
    x_train = x_train.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)

    full_ds = _MNISTDataset(x_train, y_train)
    train_len = int(0.9 * len(full_ds))
    val_len = len(full_ds) - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    ds = train_ds if split == "train" else val_ds
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config):
    model.train()
    images, labels = batch
    optimizer.zero_grad()
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    return loss.item()
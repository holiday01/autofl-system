# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity

_TRAIN_VAL_SEED = 42
_TRAIN_FRACTION = 0.8


class IXIGenderDataset(Dataset):
    def __init__(self, image_files, labels, transform):
        self._ds = ImageDataset(image_files=image_files, labels=labels, transform=transform)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


def _get_split_indices(n_total):
    n_train = max(1, int(n_total * _TRAIN_FRACTION))
    n_val = n_total - n_train
    dummy = torch.utils.data.TensorDataset(torch.zeros(n_total))
    gen = torch.Generator().manual_seed(_TRAIN_VAL_SEED)
    train_sub, val_sub = random_split(dummy, [n_train, n_val], generator=gen)
    return train_sub.indices, val_sub.indices


def build_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(
        spatial_dims=3, in_channels=1, out_channels=2
    ).to(device)
    return model


def build_dataloader(config, split="train"):
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)

    train_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()]
    )
    val_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))]
    )

    images = sorted(glob.glob(os.path.join(data_path, "*.nii.gz")))
    labels = np.load(os.path.join(data_path, "labels.npy"))

    train_indices, val_indices = _get_split_indices(len(images))

    if split == "train":
        ds = IXIGenderDataset(
            image_files=[images[i] for i in train_indices],
            labels=labels[train_indices],
            transform=train_transforms,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        ds = IXIGenderDataset(
            image_files=[images[i] for i in val_indices],
            labels=labels[val_indices],
            transform=val_transforms,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )


def train_step(model, batch, optimizer, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.CrossEntropyLoss()

    inputs, labels = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
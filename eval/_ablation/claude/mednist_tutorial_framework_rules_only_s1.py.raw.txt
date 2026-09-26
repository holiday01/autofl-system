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

import logging
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, LoadImage, RandRotate90, Resize, ScaleIntensity


_IMAGES = [
    "IXI314-IOP-0889-T1.nii.gz",
    "IXI249-Guys-1072-T1.nii.gz",
    "IXI609-HH-2600-T1.nii.gz",
    "IXI173-HH-1590-T1.nii.gz",
    "IXI020-Guys-0700-T1.nii.gz",
    "IXI342-Guys-0909-T1.nii.gz",
    "IXI134-Guys-0780-T1.nii.gz",
    "IXI577-HH-2661-T1.nii.gz",
    "IXI066-Guys-0731-T1.nii.gz",
    "IXI130-HH-1528-T1.nii.gz",
    "IXI607-Guys-1097-T1.nii.gz",
    "IXI175-HH-1570-T1.nii.gz",
    "IXI385-HH-2078-T1.nii.gz",
    "IXI344-Guys-0905-T1.nii.gz",
    "IXI409-Guys-0960-T1.nii.gz",
    "IXI584-Guys-1129-T1.nii.gz",
    "IXI253-HH-1694-T1.nii.gz",
    "IXI092-HH-1436-T1.nii.gz",
    "IXI574-IOP-1156-T1.nii.gz",
    "IXI585-Guys-1130-T1.nii.gz",
]

_LABELS = np.array(
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64
)

_train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
_val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])


class _IXIBase(torch.utils.data.Dataset):
    def __init__(self, image_files, labels):
        self.image_files = image_files
        self.labels = labels
        self._loader = LoadImage(image_only=True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = self._loader(self.image_files[idx])
        return img, self.labels[idx]


class _TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)


def build_dataloader(config, split="train"):
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)

    image_files = [os.path.join(data_path, f) for f in _IMAGES]

    base_ds = _IXIBase(image_files, _LABELS)
    n_train = int(0.8 * len(base_ds))
    n_val = len(base_ds) - n_train
    train_subset, val_subset = torch.utils.data.random_split(base_ds, [n_train, n_val])

    if split == "train":
        ds = _TransformDataset(train_subset, _train_transforms)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
    ds = _TransformDataset(val_subset, _val_transforms)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def train_step(model, batch, optimizer, config):
    device = next(model.parameters()).device
    loss_function = torch.nn.CrossEntropyLoss()
    inputs, labels = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss
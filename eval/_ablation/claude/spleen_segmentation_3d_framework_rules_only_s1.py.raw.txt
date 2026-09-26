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
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)
from monai.visualize import plot_2d_or_3d_image


class _FileList(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]


def build_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return model


def build_dataloader(config, split="train"):
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)

    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))
    all_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    n_total = len(all_files)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    file_ds = _FileList(all_files)
    train_file_ds, val_file_ds = random_split(file_ds, [n_train, n_val])
    train_files = [all_files[i] for i in train_file_ds.indices]
    val_files = [all_files[i] for i in val_file_ds.indices]

    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"],
                label_key="seg",
                spatial_size=[96, 96, 96],
                pos=1,
                neg=1,
                num_samples=4,
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            ScaleIntensityd(keys="img"),
        ]
    )

    if split == "train":
        ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        ds = monai.data.Dataset(data=val_files, transform=val_transforms)
        return DataLoader(
            ds,
            batch_size=1,
            num_workers=4,
            collate_fn=list_data_collate,
        )


def train_step(model, batch, optimizer, config):
    device = next(model.parameters()).device
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    inputs = batch["img"].to(device)
    labels = batch["seg"].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
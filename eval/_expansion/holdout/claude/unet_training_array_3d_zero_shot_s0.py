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

from glob import glob

import torch
import monai
from monai.data import ImageDataset, DataLoader
from monai.losses import DiceLoss
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirst,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)


def build_model(config):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = monai.networks.nets.UNet(
        spatial_dims=config.get("spatial_dims", 3),
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        channels=config.get("channels", (16, 32, 64, 128, 256)),
        strides=config.get("strides", (2, 2, 2, 2)),
        num_res_units=config.get("num_res_units", 2),
    ).to(device)
    return model


def build_dataloader(config, split):
    data_dir = config["data_dir"]
    images = sorted(glob(f"{data_dir}/im*.nii.gz"))
    segs = sorted(glob(f"{data_dir}/seg*.nii.gz"))

    train_imtrans = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        RandSpatialCrop((96, 96, 96), random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 2)),
    ])
    train_segtrans = Compose([
        EnsureChannelFirst(),
        RandSpatialCrop((96, 96, 96), random_size=False),
        RandRotate90(prob=0.5, spatial_axes=(0, 2)),
    ])
    val_imtrans = Compose([ScaleIntensity(), EnsureChannelFirst()])
    val_segtrans = Compose([EnsureChannelFirst()])

    n_train = config.get("n_train", len(images) // 2)

    if split == "train":
        ds = ImageDataset(
            images[:n_train], segs[:n_train],
            transform=train_imtrans, seg_transform=train_segtrans,
        )
        return DataLoader(
            ds,
            batch_size=config.get("batch_size", 4),
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=torch.cuda.is_available(),
        )
    elif split == "val":
        ds = ImageDataset(
            images[n_train:], segs[n_train:],
            transform=val_imtrans, seg_transform=val_segtrans,
        )
        return DataLoader(
            ds,
            batch_size=config.get("val_batch_size", 1),
            num_workers=config.get("num_workers", 4),
            pin_memory=torch.cuda.is_available(),
        )
    else:
        raise ValueError(f"Unknown split: {split!r}. Expected 'train' or 'val'.")


def train_step(model, batch, optimizer, config):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    loss_function = DiceLoss(sigmoid=True)

    inputs, labels = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss.item()
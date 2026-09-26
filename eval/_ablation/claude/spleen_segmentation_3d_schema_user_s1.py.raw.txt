import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

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


class _SyntheticSegDataset(torch.utils.data.Dataset):
    def __init__(self, size: int, spatial_size=(96, 96, 96)):
        self.size = size
        self.spatial_size = spatial_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.randn(1, *self.spatial_size)
        seg = torch.randint(0, 2, (1, *self.spatial_size)).float()
        return {"img": img, "seg": seg}


class _FileListDataset(torch.utils.data.Dataset):
    def __init__(self, file_dicts):
        self.file_dicts = file_dicts

    def __len__(self):
        return len(self.file_dicts)

    def __getitem__(self, idx):
        return self.file_dicts[idx]


def build_model(config: dict) -> torch.nn.Module:
    kwargs = config.get("model_kwargs", {})
    return monai.networks.nets.UNet(
        spatial_dims=kwargs.get("spatial_dims", 3),
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 1),
        channels=kwargs.get("channels", (16, 32, 64, 128, 256)),
        strides=kwargs.get("strides", (2, 2, 2, 2)),
        num_res_units=kwargs.get("num_res_units", 2),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

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

    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if images and segs:
        all_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
        n_total = len(all_files)
        n_train = max(1, int(0.8 * n_total))
        n_val = max(1, n_total - n_train)
        if n_train + n_val > n_total:
            n_val = n_total - n_train

        file_index_ds = _FileListDataset(all_files)
        train_sub, val_sub = random_split(file_index_ds, [n_train, n_val])
        train_files = [all_files[i] for i in train_sub.indices]
        val_files = [all_files[i] for i in val_sub.indices]

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
                batch_size=batch_size,
                num_workers=4,
                collate_fn=list_data_collate,
            )

    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"No NIfTI image/segmentation pairs (img*.nii.gz / seg*.nii.gz) found at "
            f"data_path='{data_path}'. Set config['allow_synthetic_data']=True to use "
            "randomly generated tensors instead."
        )

    n_synthetic = config.get("local", {}).get("synthetic_size", 40)
    full_ds = _SyntheticSegDataset(size=n_synthetic)
    n_train = max(1, int(0.8 * n_synthetic))
    n_val = max(1, n_synthetic - n_train)
    train_sub, val_sub = random_split(full_ds, [n_train, n_val])

    if split == "train":
        return DataLoader(
            train_sub,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        return DataLoader(val_sub, batch_size=batch_size)


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    inputs = batch["img"].to(device)
    labels = batch["seg"].to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    outputs = model(inputs)
    return loss_function(outputs, labels)
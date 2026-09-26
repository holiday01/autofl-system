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
from torch.utils.data import DataLoader, Dataset, random_split
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _FileListDataset(Dataset):
    """Thin wrapper around a list of file-path dicts so that random_split
    can be applied to obtain reproducible train/val index partitions."""

    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.file_list[idx]


class _SyntheticSegDataset(Dataset):
    """Purely in-memory synthetic 3-D segmentation dataset.
    Only used when config['allow_synthetic_data'] is explicitly True."""

    def __init__(self, length: int, spatial_size=(96, 96, 96)):
        self.length = length
        self.spatial_size = tuple(spatial_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(1, *self.spatial_size)
        seg = torch.randint(0, 2, (1, *self.spatial_size)).float()
        return {"img": img, "seg": seg}


def _make_transforms():
    """Return (train_transforms, val_transforms) as MONAI Compose objects."""
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
    return train_transforms, val_transforms


# ---------------------------------------------------------------------------
# FL interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the 3-D segmentation UNet.

    Supported model_kwargs keys (all optional, fall back to original defaults):
        spatial_dims, in_channels, out_channels, channels, strides, num_res_units
    """
    kw = config.get("model_kwargs", {})
    model = monai.networks.nets.UNet(
        spatial_dims=kw.get("spatial_dims", 3),
        in_channels=kw.get("in_channels", 1),
        out_channels=kw.get("out_channels", 1),
        channels=tuple(kw.get("channels", (16, 32, 64, 128, 256))),
        strides=tuple(kw.get("strides", (2, 2, 2, 2))),
        num_res_units=kw.get("num_res_units", 2),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Config keys consumed:
        data_path              (str)   root directory containing img*.nii.gz / seg*.nii.gz
        local.batch_size       (int)   mini-batch size (default 16)
        val_frac               (float) fraction reserved for validation (default 0.2)
        seed                   (int)   RNG seed for the random split (default 42)
        allow_synthetic_data   (bool)  must be True to permit synthetic fallback
        synthetic_samples      (int)   number of synthetic volumes (default 40)
        model_kwargs.spatial_size      spatial shape for synthetic crops (default [96,96,96])
    """
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)
    val_frac = config.get("val_frac", 0.2)
    seed = config.get("seed", 42)
    generator = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------
    # Attempt to discover real NIfTI data
    # ------------------------------------------------------------------
    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))
    have_real_data = bool(images and segs and len(images) == len(segs))

    if not have_real_data:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No matching NIfTI pairs (img*.nii.gz / seg*.nii.gz) found in "
                f"'{data_path}'. Provide real data at that path, or set "
                "config['allow_synthetic_data']=True to use in-memory synthetic "
                "volumes for testing purposes only."
            )

        # ------------------------------------------------------------------
        # Synthetic fallback (gated)
        # ------------------------------------------------------------------
        n_total = config.get("synthetic_samples", 40)
        n_val = max(1, int(n_total * val_frac))
        n_train = n_total - n_val
        spatial_size = tuple(
            config.get("model_kwargs", {}).get("spatial_size", [96, 96, 96])
        )
        full_ds = _SyntheticSegDataset(n_total, spatial_size=spatial_size)
        train_subset, val_subset = random_split(
            full_ds, [n_train, n_val], generator=generator
        )
        chosen = train_subset if split == "train" else val_subset
        return DataLoader(
            chosen,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------
    # Real data: split file-path list then attach MONAI transforms
    # ------------------------------------------------------------------
    all_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
    n_total = len(all_files)
    n_val = max(1, int(n_total * val_frac))
    n_train = n_total - n_val

    # random_split on a lightweight wrapper dataset to obtain stable index splits
    index_ds = _FileListDataset(all_files)
    train_subset, val_subset = random_split(
        index_ds, [n_train, n_val], generator=generator
    )
    train_files = [all_files[i] for i in train_subset.indices]
    val_files = [all_files[i] for i in val_subset.indices]

    train_transforms, val_transforms = _make_transforms()

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
            batch_size=max(1, batch_size // 2),
            shuffle=False,
            num_workers=4,
            collate_fn=list_data_collate,
        )


def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for loss.backward() and optimizer.step().
    This function does NOT call either.
    """
    device = next(model.parameters()).device
    inputs = batch["img"].to(device)
    labels = batch["seg"].to(device)

    loss_fn = monai.losses.DiceLoss(sigmoid=True)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    # grad is attached; backward/step handled externally by the FL runtime
    return loss
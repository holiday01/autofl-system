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

"""
Auto-generated FL client module.
Original script: MONAI 3-D UNet segmentation training

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""

import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
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


# ── Transforms ────────────────────────────────────────────────────────────────

_TRAIN_TRANSFORMS = Compose([
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
])

_VAL_TRANSFORMS = Compose([
    LoadImaged(keys=["img", "seg"]),
    EnsureChannelFirstd(keys=["img", "seg"]),
    ScaleIntensityd(keys="img"),
])


# ── Datasets ──────────────────────────────────────────────────────────────────

class NIfTISegDataset(Dataset):
    """
    Wraps MONAI transforms for 3-D NIfTI image/segmentation file pairs.
    Each item is a dict {"img": Tensor, "seg": Tensor}.
    Train split applies augmentation (RandCrop + RandRotate90);
    val split applies deterministic pre-processing only.
    """

    def __init__(self, file_pairs: list, split: str = "train"):
        transforms = _TRAIN_TRANSFORMS if split == "train" else _VAL_TRANSFORMS
        self._ds = monai.data.Dataset(data=file_pairs, transform=transforms)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


class SyntheticSegDataset(Dataset):
    """
    Fully in-memory synthetic 3-D segmentation dataset.
    Returns dict {"img": float Tensor [1,D,H,W], "seg": float Tensor [1,D,H,W]}.
    For unit / smoke tests only — never used when real data is available.
    """

    def __init__(self, n: int = 40, spatial_size: tuple = (96, 96, 96)):
        self.n = n
        self.spatial_size = spatial_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.randn(1, *self.spatial_size)
        seg = torch.randint(0, 2, (1, *self.spatial_size)).float()
        return {"img": img, "seg": seg}


# ── FL Interface ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    defaults = dict(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    defaults.update(kwargs)
    return monai.networks.nets.UNet(**defaults)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    # Original script uses batch_size=2 for 3-D volumes; honour config override.
    batch_size  = local.get("batch_size", config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory", True)
    data_path   = config.get("data_path", ".")
    val_ratio   = config.get("val_ratio", 0.2)
    seed        = config.get("seed", 42)

    # ── discover real NIfTI pairs ─────────────────────────────────────────
    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs   = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))
    file_pairs = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    if file_pairs:
        # Split the file-pair list first so each subset gets the correct
        # transform pipeline (train augmentation vs. val-only pre-processing).
        # This preserves random_split semantics via a seeded index permutation.
        n_total = len(file_pairs)
        n_val   = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        indices = torch.randperm(
            n_total,
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        train_pairs = [file_pairs[i] for i in indices[:n_train]]
        val_pairs   = [file_pairs[i] for i in indices[n_train:]]
        ds = NIfTISegDataset(
            file_pairs=train_pairs if split == "train" else val_pairs,
            split=split,
        )
    else:
        # ── no real data: check the synthetic-data gate ───────────────────
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No NIfTI file pairs (img*.nii.gz / seg*.nii.gz) found under "
                f"'{data_path}'. Provide real data or set "
                f"config['allow_synthetic_data'] = True to use in-memory "
                f"synthetic tensors for smoke-testing."
            )
        full_ds = SyntheticSegDataset(
            n=config.get("synthetic_n", 40),
            spatial_size=tuple(config.get("synthetic_spatial_size", [96, 96, 96])),
        )
        n_val   = max(1, int(len(full_ds) * val_ratio))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        ds = train_ds if split == "train" else val_ds

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass. Returns the raw DiceLoss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, dict):
        inputs  = batch["img"].to(device)
        targets = batch["seg"].to(device)
    elif isinstance(batch, (list, tuple)):
        inputs  = batch[0].to(device)
        targets = batch[1].to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs   = model(inputs)
    criterion = monai.losses.DiceLoss(sigmoid=True)
    loss      = criterion(outputs, targets)
    return loss
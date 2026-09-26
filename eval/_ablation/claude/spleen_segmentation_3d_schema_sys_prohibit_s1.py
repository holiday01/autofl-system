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
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split
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
# Internal dataset helpers
# ---------------------------------------------------------------------------

class _NiftiFileDataset(TorchDataset):
    """Stores raw file-path dicts; transforms are applied by the wrapper below.

    Keeping path storage and transform application separate lets us call
    random_split on a single, transform-free dataset and then attach
    split-specific MONAI pipelines afterwards.
    """

    def __init__(self, files):
        self._files = files

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        return self._files[idx]          # {"img": <path>, "seg": <path>}


class _TransformWrapper(TorchDataset):
    """Applies a MONAI Compose pipeline to every item drawn from a Subset.

    When RandCropByPosNegLabeld (num_samples > 1) is in the pipeline the
    transform returns a list of dicts; list_data_collate handles that case.
    """

    def __init__(self, subset, transform):
        self._subset = subset
        self._transform = transform

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        sample = self._subset[idx]
        return self._transform(sample)


class _SyntheticNiftiDataset(TorchDataset):
    """Synthetic 3-D image / segmentation pairs for smoke-testing.

    Only used when the caller has explicitly set
    config['allow_synthetic_data'] = True.
    """

    def __init__(self, size: int = 40, spatial_size=(96, 96, 96)):
        self._size = size
        self._spatial_size = spatial_size

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        img = torch.randn(1, *self._spatial_size)
        seg = torch.randint(0, 2, (1, *self._spatial_size)).float()
        return {"img": img, "seg": seg}


# ---------------------------------------------------------------------------
# FL client API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate the 3-D MONAI UNet.

    All constructor knobs can be overridden via config['model_kwargs'].
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
    """Return a DataLoader for the requested split.

    Real data
    ---------
    Expects NIfTI files matching ``img*.nii.gz`` / ``seg*.nii.gz`` under
    ``config['data_path']``.  A single ``_NiftiFileDataset`` is built from
    all pairs, then ``random_split`` produces the train / val subsets, and
    split-specific MONAI transform pipelines are applied via
    ``_TransformWrapper``.

    Synthetic fallback
    ------------------
    Activated only when real files are absent **and**
    ``config['allow_synthetic_data'] is True``.  If that flag is False (or
    absent) and files are missing, a ``FileNotFoundError`` is raised.
    """
    local_cfg = config.get("local", {})
    batch_size = local_cfg.get("batch_size", 16)
    data_path = config.get("data_path", ".")

    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))
    real_data_available = bool(images and segs)

    # ------------------------------------------------------------------
    # Synthetic fallback (gated)
    # ------------------------------------------------------------------
    if not real_data_available:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No NIfTI image/segmentation files found in '{data_path}'. "
                "Provide real data or set config['allow_synthetic_data'] = True "
                "to enable the synthetic-data fallback."
            )
        full_dataset = _SyntheticNiftiDataset(size=40, spatial_size=(96, 96, 96))
        total = len(full_dataset)
        train_size = int(0.8 * total)
        val_size = total - train_size
        train_subset, val_subset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        dataset = train_subset if split == "train" else val_subset
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------
    # Real NIfTI data
    # ------------------------------------------------------------------
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

    all_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
    total = len(all_files)
    train_size = int(0.8 * total)
    val_size = total - train_size

    # Single dataset → random_split → per-split transform wrappers
    full_dataset = _NiftiFileDataset(all_files)
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    if split == "train":
        dataset = _TransformWrapper(train_subset, train_transforms)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        dataset = _TransformWrapper(val_subset, val_transforms)
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            collate_fn=list_data_collate,
        )


def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Single forward pass; returns the live loss tensor.

    The FL runtime is responsible for loss.backward() and optimizer.step().
    Do NOT call them here.
    """
    device = next(model.parameters()).device
    inputs = batch["img"].to(device)
    labels = batch["seg"].to(device)

    loss_fn = monai.losses.DiceLoss(sigmoid=True)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    return loss
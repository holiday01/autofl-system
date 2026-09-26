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
# Internal helpers
# ---------------------------------------------------------------------------

class _ListDataset(TorchDataset):
    """Minimal Dataset wrapper around a plain Python list.

    Needed so that torch.utils.data.random_split can operate on a list of
    file-path dicts before the MONAI Dataset (with its transforms) is built.
    """

    def __init__(self, items: list):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class _SyntheticDataset(TorchDataset):
    """Generates random (1, D, H, W) tensors that mimic pre-cropped 3-D volumes.

    Only instantiated when config['allow_synthetic_data'] is True.
    Spatial size defaults to the crop size used by RandCropByPosNegLabeld.
    """

    def __init__(self, length: int = 40, spatial_size: tuple = (96, 96, 96)):
        self.length = length
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(1, *self.spatial_size)
        seg = torch.randint(0, 2, (1, *self.spatial_size)).float()
        return {"img": img, "seg": seg}


# ---------------------------------------------------------------------------
# FL API — three required entry-points
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the 3-D UNet segmentation model.

    All constructor arguments are drawn from config.get("model_kwargs", {});
    the original architecture defaults are preserved as fallbacks.
    """
    kwargs = config.get("model_kwargs", {})
    model = monai.networks.nets.UNet(
        spatial_dims=kwargs.get("spatial_dims", 3),
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 1),
        channels=tuple(kwargs.get("channels", (16, 32, 64, 128, 256))),
        strides=tuple(kwargs.get("strides", (2, 2, 2, 2))),
        num_res_units=kwargs.get("num_res_units", 2),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    Real-data path
    --------------
    Looks for ``img*.nii.gz`` / ``seg*.nii.gz`` pairs under
    ``config["data_path"]`` (default ``"."``).  All pairs are pooled and then
    split 80 / 20 train / val via ``random_split`` (seed 42 for
    reproducibility across clients).

    MONAI transforms (including RandCropByPosNegLabeld with num_samples=4) are
    applied *inside* the per-split ``monai.data.Dataset``, so each split gets
    its own correct augmentation pipeline.

    Synthetic-data fallback
    -----------------------
    Only activated when ``config["allow_synthetic_data"] is True``.
    If real data are absent and the flag is ``False`` (the default), a
    ``FileNotFoundError`` is raised so the FL runtime can surface a clear
    diagnostic rather than silently training on garbage.
    """
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)

    # ------------------------------------------------------------------ #
    # MONAI transform pipelines (identical to the original script)        #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Discover real NIfTI files                                           #
    # ------------------------------------------------------------------ #
    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if not images or not segs or len(images) != len(segs):
        # ---------------------------------------------------------------- #
        # Synthetic fallback — GATED on allow_synthetic_data               #
        # ---------------------------------------------------------------- #
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No matching NIfTI file pairs (img*.nii.gz / seg*.nii.gz) found "
                f"under '{data_path}'. "
                "Provide real data or set config['allow_synthetic_data'] = True "
                "to enable the synthetic-data fallback."
            )

        logging.warning(
            "Real NIfTI data not found under '%s'. "
            "Falling back to synthetic random tensors "
            "(allow_synthetic_data=True).",
            data_path,
        )
        n_total = 40
        n_val = max(1, int(n_total * 0.2))
        n_train = n_total - n_val
        full_syn = _SyntheticDataset(length=n_total, spatial_size=(96, 96, 96))
        train_sub, val_sub = random_split(
            full_syn,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        ds = train_sub if split == "train" else val_sub
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------ #
    # Split real file pairs with random_split                             #
    # ------------------------------------------------------------------ #
    all_files = [
        {"img": img, "seg": seg} for img, seg in zip(images, segs)
    ]
    n_total = len(all_files)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    # random_split needs a Dataset; wrap the list, split, then recover indices.
    list_ds = _ListDataset(all_files)
    train_subset, val_subset = random_split(
        list_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    if split == "train":
        file_list = [all_files[i] for i in train_subset.indices]
        transforms = train_transforms
        shuffle = True
    else:
        file_list = [all_files[i] for i in val_subset.indices]
        transforms = val_transforms
        shuffle = False

    # monai.data.Dataset applies the transform pipeline per item;
    # list_data_collate is required because RandCropByPosNegLabeld returns
    # a *list* of dicts (num_samples crops) per image rather than a single dict.
    ds = monai.data.Dataset(data=file_list, transform=transforms)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run ONE forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.
    """
    device = next(model.parameters()).device

    inputs = batch["img"].to(device)
    labels = batch["seg"].to(device)

    outputs = model(inputs)

    loss_fn = monai.losses.DiceLoss(sigmoid=True)
    loss = loss_fn(outputs, labels)

    # Return loss with grad attached — do NOT call backward() or step().
    return loss
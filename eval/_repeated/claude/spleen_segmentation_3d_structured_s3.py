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
# Helper datasets (module-level so they are picklable for num_workers > 0)
# ---------------------------------------------------------------------------

class _SyntheticSegDataset(TorchDataset):
    """Synthetic dict-returning dataset compatible with MONAI DataLoader collation."""

    def __init__(self, size: int = 40, spatial_size: tuple = (96, 96, 96)):
        self.size = size
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        img = torch.randn(1, *self.spatial_size)
        seg = torch.randint(0, 2, (1, *self.spatial_size)).float()
        return {"img": img, "seg": seg}


class _IndexDataset(TorchDataset):
    """Thin wrapper over a range so random_split can produce reproducible index subsets."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate the MONAI 3-D UNet.

    All constructor arguments can be overridden through
    ``config["model_kwargs"]``.
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
    """Return a DataLoader for *split* ("train" or "val").

    Expects NIfTI pairs ``img*.nii.gz`` / ``seg*.nii.gz`` under
    ``config["data_path"]``.  When those files are absent the function
    raises ``FileNotFoundError`` unless ``config["allow_synthetic_data"]``
    is ``True``, in which case it falls back to random tensors.

    The train/val split is always produced via ``random_split`` from a
    single combined dataset (80 % train / 20 % val).
    """
    local_cfg = config.get("local", {})
    batch_size = local_cfg.get("batch_size", 16)
    data_path = config.get("data_path", ".")

    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))
    have_real_data = bool(images and segs)

    # ── Synthetic fallback ───────────────────────────────────────────────────
    if not have_real_data:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No NIfTI files matching 'img*.nii.gz' / 'seg*.nii.gz' were found "
                f"under '{data_path}'. Either provide real data at that path or set "
                "config['allow_synthetic_data'] = True to use synthetic tensors."
            )
        full_ds = _SyntheticSegDataset(size=40, spatial_size=(96, 96, 96))
        n_total = len(full_ds)
        n_val = max(1, int(n_total * 0.2))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        chosen_ds = train_ds if split == "train" else val_ds
        return DataLoader(
            chosen_ds,
            batch_size=batch_size if split == "train" else 1,
            shuffle=(split == "train"),
            num_workers=0,            # Subset + inner class; keep workers=0
            pin_memory=torch.cuda.is_available(),
        )

    # ── Real NIfTI path ──────────────────────────────────────────────────────
    all_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
    n_total = len(all_files)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    # random_split on a lightweight index dataset → reproducible file-list split
    train_idx_subset, val_idx_subset = random_split(
        _IndexDataset(n_total), [n_train, n_val]
    )
    train_files = [all_files[i] for i in sorted(train_idx_subset.indices)]
    val_files = [all_files[i] for i in sorted(val_idx_subset.indices)]

    # MONAI transforms (identical to the original script)
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


def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime calls ``loss.backward()`` and ``optimizer.step()``
    externally; this function must NOT do either.
    """
    device = next(model.parameters()).device
    inputs = batch["img"].to(device)
    labels = batch["seg"].to(device)
    loss_fn = monai.losses.DiceLoss(sigmoid=True)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    return loss
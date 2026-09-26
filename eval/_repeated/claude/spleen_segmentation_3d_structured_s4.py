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


# ── preserved model defaults ──────────────────────────────────────────────────

_DEFAULT_UNET_KWARGS = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# Module-level loss instance; DiceLoss is stateless between calls.
_loss_fn = monai.losses.DiceLoss(sigmoid=True)


# ── helper dataset wrappers ───────────────────────────────────────────────────

class _FileListDataset(Dataset):
    """Returns raw file-path dicts so random_split can operate on the full list
    before any (potentially split-specific) MONAI transforms are applied."""

    def __init__(self, file_dicts):
        self.file_dicts = file_dicts

    def __len__(self):
        return len(self.file_dicts)

    def __getitem__(self, idx):
        return self.file_dicts[idx]


class _MonaiTransformDataset(Dataset):
    """Applies a MONAI Compose pipeline on top of a random_split Subset.

    Needed so train and val subsets can carry different augmentation pipelines
    while both originating from the same random_split call.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # self.subset[idx] returns a raw file-path dict; transform loads + augs it.
        return self.transform(self.subset[idx])


class _SyntheticSegDataset(Dataset):
    """Synthetic 3-D segmentation dataset with the same {img, seg} dict format.

    Only instantiated when config['allow_synthetic_data'] is True.
    """

    def __init__(self, length: int = 40, spatial_size=(128, 128, 128)):
        self.length = length
        self.spatial_size = spatial_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(1, *self.spatial_size)
        seg = torch.randint(0, 2, (1, *self.spatial_size)).float()
        return {"img": img, "seg": seg}


# ── FL interface ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the 3-D UNet.

    Caller may override any constructor argument via config['model_kwargs'].
    JSON-serialised lists for 'channels' and 'strides' are coerced to tuples.
    """
    kwargs = {**_DEFAULT_UNET_KWARGS, **config.get("model_kwargs", {})}
    # JSON round-trips deliver lists; UNet expects tuples.
    kwargs["channels"] = tuple(kwargs["channels"])
    kwargs["strides"] = tuple(kwargs["strides"])
    return monai.networks.nets.UNet(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Real data layout expected under config['data_path']:
        img0.nii.gz, img1.nii.gz, …   (images)
        seg0.nii.gz, seg1.nii.gz, …   (segmentation masks)

    The full file list is split 80/20 train/val via random_split so both
    splits are drawn from the same shuffled pool.

    If no files are found and config['allow_synthetic_data'] is False
    (the default), FileNotFoundError is raised.  Only when that flag is
    explicitly True will synthetic tensors be used.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path  = config.get("data_path", ".")

    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs   = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    # ── synthetic fallback (explicitly opt-in only) ───────────────────────────
    if not images or not segs:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No NIfTI files matching 'img*.nii.gz' / 'seg*.nii.gz' were "
                f"found in '{data_path}'. Supply real data or set "
                "config['allow_synthetic_data']=True to enable the synthetic "
                "tensor fallback."
            )
        synth_len = config.get("local", {}).get("synthetic_length", 40)
        full_ds = _SyntheticSegDataset(length=synth_len)
        total   = len(full_ds)
        n_train = max(1, min(total - 1, int(total * 0.8)))
        n_val   = total - n_train
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        chosen_ds = train_ds if split == "train" else val_ds
        return DataLoader(
            chosen_ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=4,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )

    # ── real-data path ────────────────────────────────────────────────────────
    file_dicts = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    # random_split on the raw file-dict list (no I/O yet).
    base_ds = _FileListDataset(file_dicts)
    total   = len(base_ds)
    n_train = max(1, min(total - 1, int(total * 0.8)))
    n_val   = total - n_train
    train_subset, val_subset = random_split(base_ds, [n_train, n_val])

    # MONAI transforms — identical to the original script.
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
        chosen_ds = _MonaiTransformDataset(train_subset, train_transforms)
    else:
        chosen_ds = _MonaiTransformDataset(val_subset, val_transforms)

    return DataLoader(
        chosen_ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        collate_fn=list_data_collate,   # handles per-sample lists from RandCrop
        pin_memory=torch.cuda.is_available(),
    )


def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Single forward pass.

    Returns the loss tensor with its computation graph intact.
    The FL runtime calls loss.backward() and optimizer.step(); do NOT do so here.
    """
    device = next(model.parameters()).device
    inputs = batch["img"].to(device)
    labels = batch["seg"].to(device)
    outputs = model(inputs)
    loss = _loss_fn(outputs, labels)
    return loss
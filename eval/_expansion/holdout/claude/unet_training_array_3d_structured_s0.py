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
from monai.data import ImageDataset, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
from monai.visualize import plot_2d_or_3d_image


# ── helper datasets ───────────────────────────────────────────────────────────

class _RawNiftiSegDataset(torch.utils.data.Dataset):
    """Loads NIfTI image+seg pairs as raw float32 arrays (no transforms applied).

    A single instance of this class is created over all available files;
    random_split then carves out train/val subsets, after which
    _TransformSubset overlays the appropriate MONAI transform pipeline.
    """

    def __init__(self, image_paths, seg_paths):
        assert len(image_paths) == len(seg_paths), (
            f"Mismatch: {len(image_paths)} images vs {len(seg_paths)} segs"
        )
        self.image_paths = list(image_paths)
        self.seg_paths = list(seg_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        seg = nib.load(self.seg_paths[idx]).get_fdata().astype(np.float32)
        return img, seg


class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a torch Subset and applies separate MONAI transforms to image/seg."""

    def __init__(self, subset, img_transform=None, seg_transform=None):
        self.subset = subset
        self.img_transform = img_transform
        self.seg_transform = seg_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, seg = self.subset[idx]
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.seg_transform is not None:
            seg = self.seg_transform(seg)
        return img, seg


class _SyntheticSegDataset(torch.utils.data.Dataset):
    """Fully in-memory synthetic 3-D segmentation dataset.

    Only ever constructed when config['allow_synthetic_data'] is True.
    Items are already channel-first float tensors, so no further transforms
    are needed (the synthetic path intentionally skips augmentation).
    """

    def __init__(self, length=40, spatial_size=(96, 96, 96)):
        self.length = length
        self.spatial_size = tuple(spatial_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(1, *self.spatial_size)
        seg = torch.randint(0, 2, (1, *self.spatial_size)).float()
        return img, seg


# ── FL API ────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the MONAI 3-D UNet.

    All constructor arguments can be overridden via config['model_kwargs'].
    Defaults mirror the original training script exactly.
    """
    kwargs = config.get("model_kwargs", {})

    # tuple() ensures JSON lists (which are lists, not tuples) are accepted
    channels = tuple(kwargs.get("channels", (16, 32, 64, 128, 256)))
    strides  = tuple(kwargs.get("strides",  (2, 2, 2, 2)))

    model = monai.networks.nets.UNet(
        spatial_dims=kwargs.get("spatial_dims", 3),
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 1),
        channels=channels,
        strides=strides,
        num_res_units=kwargs.get("num_res_units", 2),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Real data discovery
    -------------------
    Scans config['data_path'] (default '.') for files matching
    ``im*.nii.gz`` (images) and ``seg*.nii.gz`` (segmentation masks).
    A single _RawNiftiSegDataset is built over *all* found files and then
    split with torch.utils.data.random_split according to config['val_frac']
    (default 0.2).  The train subset receives augmentation transforms;
    the val subset receives inference-only transforms.

    Synthetic fallback
    ------------------
    If no real files are found and config['allow_synthetic_data'] is True,
    a _SyntheticSegDataset is used instead.  If that flag is False (default),
    a FileNotFoundError is raised immediately — synthetic data is never used
    silently.
    """
    local_cfg   = config.get("local", {})
    batch_size  = local_cfg.get("batch_size", 16)
    num_workers = local_cfg.get("num_workers", 2)
    data_path   = config.get("data_path", ".")
    val_frac    = float(config.get("val_frac", 0.2))
    seed        = int(config.get("seed", 42))

    # ── MONAI transform pipelines (identical to original script) ─────────────
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
    val_imtrans  = Compose([ScaleIntensity(), EnsureChannelFirst()])
    val_segtrans = Compose([EnsureChannelFirst()])

    # ── discover real NIfTI files ─────────────────────────────────────────────
    images = sorted(glob(os.path.join(data_path, "im*.nii.gz")))
    segs   = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))
    have_real_data = len(images) > 0 and len(segs) > 0

    if not have_real_data:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No NIfTI files matching 'im*.nii.gz' / 'seg*.nii.gz' were "
                f"found under data_path='{data_path}'. "
                "Either point config['data_path'] at your dataset directory or "
                "set config['allow_synthetic_data'] = True to use synthetic "
                "data for testing/CI purposes."
            )
        # ── synthetic branch (CI / smoke-test only) ───────────────────────────
        syn_len  = int(config.get("synthetic_length", 40))
        syn_size = config.get("synthetic_spatial_size", [96, 96, 96])
        full_ds  = _SyntheticSegDataset(length=syn_len, spatial_size=syn_size)
        n_val    = max(1, int(len(full_ds) * val_frac))
        n_train  = len(full_ds) - n_val
        train_sub, val_sub = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        subset  = train_sub if split == "train" else val_sub
        shuffle = split == "train"
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # ── real-data branch ──────────────────────────────────────────────────────
    raw_ds  = _RawNiftiSegDataset(images, segs)
    n_val   = max(1, int(len(raw_ds) * val_frac))
    n_train = len(raw_ds) - n_val
    train_sub, val_sub = random_split(
        raw_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    if split == "train":
        dataset = _TransformSubset(
            train_sub,
            img_transform=train_imtrans,
            seg_transform=train_segtrans,
        )
        shuffle = True
    else:
        dataset = _TransformSubset(
            val_sub,
            img_transform=val_imtrans,
            seg_transform=val_segtrans,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for loss.backward() and optimizer.step().
    This function must NOT call either of those.

    Parameters
    ----------
    model     : the UNet returned by build_model, already on the target device.
    batch     : (inputs, labels) tuple as yielded by build_dataloader.
    optimizer : provided by the FL runtime (not stepped here).
    config    : runtime configuration dict (unused beyond device resolution).

    Returns
    -------
    loss : scalar Tensor, grad_fn attached, on the model's device.
    """
    device = next(model.parameters()).device

    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    loss_fn = monai.losses.DiceLoss(sigmoid=True)

    outputs = model(inputs)
    loss    = loss_fn(outputs, labels)
    # loss.backward() and optimizer.step() are intentionally omitted;
    # the FL runtime handles gradient aggregation and weight updates.
    return loss
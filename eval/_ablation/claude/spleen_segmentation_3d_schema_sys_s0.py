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
# Internal helper: a minimal torch Dataset that wraps a list of dicts so that
# random_split (which requires a Dataset) can be used to split the file list.
# ---------------------------------------------------------------------------
class _FileListDataset(torch.utils.data.Dataset):
    """Thin wrapper around a list of file-dict entries for random_split."""

    def __init__(self, file_dicts):
        self.file_dicts = file_dicts

    def __len__(self):
        return len(self.file_dicts)

    def __getitem__(self, idx):
        return self.file_dicts[idx]


# ---------------------------------------------------------------------------
# 1.  build_model
# ---------------------------------------------------------------------------
def build_model(config: dict) -> torch.nn.Module:
    """Instantiate the 3-D MONAI UNet and return it (no device placement)."""
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


# ---------------------------------------------------------------------------
# 2.  build_dataloader
# ---------------------------------------------------------------------------
def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Data discovery
    --------------
    Looks for ``img*.nii.gz`` / ``seg*.nii.gz`` pairs under
    ``config['data_path']`` (default: ``'.'``).

    Synthetic-data fallback
    -----------------------
    If no files are found AND ``config['allow_synthetic_data']`` is ``True``,
    synthetic NIfTI volumes are written to a temporary directory and used
    instead.  If the flag is ``False`` (the default), a ``FileNotFoundError``
    is raised immediately — synthetic data is *never* used silently.

    Train / Val split
    -----------------
    All discovered files are wrapped in a single :class:`_FileListDataset`
    and split via :func:`torch.utils.data.random_split` (80 % train /
    20 % val, seeded by ``config.get('seed', 42)``).

    MONAI transforms
    ----------------
    * **train** — LoadImaged → EnsureChannelFirstd → ScaleIntensityd →
      RandCropByPosNegLabeld → RandRotate90d
    * **val**   — LoadImaged → EnsureChannelFirstd → ScaleIntensityd
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    seed = config.get("seed", 42)

    # ------------------------------------------------------------------
    # Discover NIfTI file pairs
    # ------------------------------------------------------------------
    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if len(images) == 0 or len(segs) == 0:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No NIfTI image/segmentation files (img*.nii.gz / seg*.nii.gz) "
                f"found under data_path='{data_path}'. "
                "Either point 'data_path' at your dataset or set "
                "config['allow_synthetic_data']=True to use synthetic volumes "
                "for smoke-testing."
            )

        # --- Synthetic fallback (only reached when explicitly allowed) ---
        n_synthetic = config.get("synthetic_samples", 20)
        _syn_dir = tempfile.mkdtemp(prefix="fl_monai_synthetic_")
        logging.getLogger(__name__).warning(
            "allow_synthetic_data=True: generating %d synthetic volumes in %s",
            n_synthetic,
            _syn_dir,
        )
        for i in range(n_synthetic):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            nib.save(
                nib.Nifti1Image(im, np.eye(4)),
                os.path.join(_syn_dir, f"img{i:d}.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(seg, np.eye(4)),
                os.path.join(_syn_dir, f"seg{i:d}.nii.gz"),
            )
        images = sorted(glob(os.path.join(_syn_dir, "img*.nii.gz")))
        segs = sorted(glob(os.path.join(_syn_dir, "seg*.nii.gz")))

    # Pair images ↔ segmentations
    all_files = [
        {"img": img, "seg": seg}
        for img, seg in zip(images, segs)
    ]

    # ------------------------------------------------------------------
    # random_split  →  train / val file lists
    # ------------------------------------------------------------------
    n_total = len(all_files)
    n_val = max(1, int(round(n_total * 0.2)))
    n_train = n_total - n_val

    base_ds = _FileListDataset(all_files)
    train_subset, val_subset = random_split(
        base_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # Materialise the subsets back into plain lists so that monai.data.Dataset
    # can wrap them with its own caching / transform logic.
    train_files = [train_subset[i] for i in range(len(train_subset))]
    val_files = [val_subset[i] for i in range(len(val_subset))]

    # ------------------------------------------------------------------
    # MONAI transforms
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

    # ------------------------------------------------------------------
    # Build the requested DataLoader
    # ------------------------------------------------------------------
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
    else:  # "val"
        ds = monai.data.Dataset(data=val_files, transform=val_transforms)
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=list_data_collate,
        )


# ---------------------------------------------------------------------------
# 3.  train_step
# ---------------------------------------------------------------------------
def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Single forward pass.

    Moves inputs / labels to the device inferred from model parameters,
    computes MONAI DiceLoss(sigmoid=True), and returns the loss tensor
    **with gradients attached**.

    The FL runtime is responsible for calling ``loss.backward()`` and
    ``optimizer.step()``; this function must NOT do either.
    """
    device = next(model.parameters()).device

    inputs = batch["img"].to(device)   # (B, 1, H, W, D)
    labels = batch["seg"].to(device)   # (B, 1, H, W, D)

    loss_fn = monai.losses.DiceLoss(sigmoid=True)

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

    # Intentionally no loss.backward() and no optimizer.step() —
    # the FL aggregation runtime owns both of those.
    return loss
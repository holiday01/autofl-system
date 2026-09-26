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
Original script: MONAI 3-D classification (DenseNet121, IXI-T1 gender classification).

Exposes:
  build_model(config)                    -> nn.Module
  build_dataloader(config, split)        -> DataLoader
  train_step(model, batch, opt, config)  -> loss tensor (with grad_fn)

CONTRACT:
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


class NIfTIClassificationDataset(Dataset):
    """
    Wraps MONAI ImageDataset for NIfTI medical image classification.
    MONAI transforms are applied inside this Dataset wrapper (satisfying the
    'wrap MONAI transforms in the Dataset class' rule).

    Supported data_path layouts:
      1. Subdirectory-per-class  — each subdir name must be int-castable
         (or is mapped to a sequential integer index):
           data_path/
             0/  *.nii.gz
             1/  *.nii.gz
      2. Flat directory with a sidecar label file:
           data_path/
             *.nii.gz
             labels.npy   (1-D int64 array, same order as sorted filenames)
             labels.csv   (one integer per line, same order)
    """

    def __init__(self, root: str, transforms: Compose):
        image_files: list[str] = []
        labels: list[int] = []

        if not os.path.isdir(root):
            raise FileNotFoundError(f"data_path directory not found: '{root}'")

        # --- Layout 1: subdirectory-per-class ---
        subdirs = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
        if subdirs:
            for rank, subdir in enumerate(subdirs):
                try:
                    label_idx = int(subdir)
                except ValueError:
                    label_idx = rank
                subpath = os.path.join(root, subdir)
                for fname in sorted(os.listdir(subpath)):
                    if fname.endswith(".nii.gz") or fname.endswith(".nii"):
                        image_files.append(os.path.join(subpath, fname))
                        labels.append(label_idx)

        # --- Layout 2: flat directory + sidecar labels ---
        if not image_files:
            nii_files = sorted(
                f for f in os.listdir(root)
                if f.endswith(".nii.gz") or f.endswith(".nii")
            )
            labels_npy = os.path.join(root, "labels.npy")
            labels_csv = os.path.join(root, "labels.csv")
            if nii_files and os.path.isfile(labels_npy):
                image_files = [os.path.join(root, f) for f in nii_files]
                labels = np.load(labels_npy).astype(np.int64).tolist()
            elif nii_files and os.path.isfile(labels_csv):
                import csv
                with open(labels_csv, newline="") as fh:
                    labels = [int(row[0]) for row in csv.reader(fh) if row]
                image_files = [os.path.join(root, f) for f in nii_files]

        if not image_files:
            raise FileNotFoundError(
                f"No NIfTI files (.nii / .nii.gz) found in '{root}'. "
                "Expected either class-labelled subdirectories or a flat directory "
                "with 'labels.npy' / 'labels.csv'."
            )

        self._monai_ds = ImageDataset(
            image_files=image_files,
            labels=np.array(labels, dtype=np.int64),
            transform=transforms,
        )

    def __len__(self) -> int:
        return len(self._monai_ds)

    def __getitem__(self, idx):
        return self._monai_ds[idx]


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    kwargs.setdefault("spatial_dims", 3)
    kwargs.setdefault("in_channels", 1)
    kwargs.setdefault("out_channels", 2)
    return monai.networks.nets.DenseNet121(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  16))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory",  True)

    spatial_size = tuple(config.get("spatial_size", (96, 96, 96)))
    data_path    = config.get("data_path", ".")
    val_ratio    = config.get("val_ratio", 0.2)
    seed         = config.get("seed", 42)

    # Build split-appropriate transforms (MONAI transforms wrapped inside Dataset)
    if split == "train":
        transforms = Compose([
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize(spatial_size),
            RandRotate90(),
        ])
    else:
        transforms = Compose([
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize(spatial_size),
        ])

    try:
        full_dataset = NIfTIClassificationDataset(root=data_path, transforms=transforms)
    except FileNotFoundError as exc:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real data not found at '{data_path}' and "
                "config['allow_synthetic_data'] is False. "
                "Point config['data_path'] to a valid NIfTI dataset directory, "
                "or set allow_synthetic_data=True for smoke-testing only."
            ) from exc
        # Synthetic fallback — gated on allow_synthetic_data
        out_channels = config.get("model_kwargs", {}).get("out_channels", 2)
        n_synthetic  = config.get("synthetic_n", 40)
        X = torch.randn(n_synthetic, 1, *spatial_size)
        y = torch.randint(0, out_channels, (n_synthetic,))
        full_dataset = TensorDataset(X, y)

    n_val   = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    ds = train_ds if split == "train" else val_ds
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch   = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs  = batch[0]
        targets = batch[1]
    elif isinstance(batch, dict):
        batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        inputs  = batch.get("input",  batch.get("x",      batch.get("image")))
        targets = batch.get("label",  batch.get("y",      batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs   = model(inputs)
    criterion = nn.CrossEntropyLoss()
    loss      = criterion(outputs, targets)
    return loss
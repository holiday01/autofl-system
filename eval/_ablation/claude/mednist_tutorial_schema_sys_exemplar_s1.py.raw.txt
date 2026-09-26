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
Original: MONAI 3-D DenseNet121 gender-classification training script.

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

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
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


# ── Dataset helpers ──────────────────────────────────────────────────────────

class _NiftiIndexDataset(Dataset):
    """
    Lightweight dataset that stores NIfTI file paths and integer labels without
    loading any image data.  Used solely as a splittable index for random_split;
    actual image I/O is handled by _TransformSubset -> MONAI ImageDataset.
    """

    def __init__(self, image_files: list, labels: np.ndarray):
        self.image_files = image_files
        self.labels = labels

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> int:
        return idx  # deferred; consumed by _TransformSubset


class _TransformSubset(Dataset):
    """
    Wraps a Subset produced by random_split, materialises a MONAI ImageDataset
    from the selected indices, and applies the requested transform pipeline.
    This lets train and val splits carry independent augmentation schedules
    while still honouring the random_split contract.
    """

    def __init__(self, subset, transform):
        files  = [subset.dataset.image_files[i] for i in subset.indices]
        labels = np.array([subset.dataset.labels[i] for i in subset.indices], dtype=np.int64)
        self.inner = ImageDataset(image_files=files, labels=labels, transform=transform)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        return self.inner[idx]


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Instantiate MONAI DenseNet121 for 3-D MRI classification.
    Default kwargs mirror the original script (spatial_dims=3, in_channels=1,
    out_channels=2).  Override any of these via config['model_kwargs'].
    """
    kwargs = config.get("model_kwargs", {})
    kwargs.setdefault("spatial_dims", 3)
    kwargs.setdefault("in_channels", 1)
    kwargs.setdefault("out_channels", 2)
    return monai.networks.nets.DenseNet121(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for a NIfTI-based binary classification dataset.

    Real-data discovery (in order):
      1. Flat layout   — <data_path>/*.nii.gz  +  <data_path>/labels.npy
      2. Class-subdir  — <data_path>/<class_N>/*.nii.gz  (label = subdir sort-index)

    Synthetic fallback:
      If no real data is found AND config['allow_synthetic_data'] is True,
      a TensorDataset of random 3-D volumes is used for testing.
      If the flag is False (the default), FileNotFoundError is raised immediately.
    """
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 16))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    data_path    = config.get("data_path", ".")
    val_ratio    = config.get("val_ratio", 0.2)
    seed         = config.get("seed", 42)
    spatial_size = tuple(config.get("spatial_size", [96, 96, 96]))

    # MONAI transform pipelines — mirror the original script exactly
    train_transforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize(spatial_size),
        RandRotate90(),
    ])
    val_transforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize(spatial_size),
    ])

    # ── Discover real data ───────────────────────────────────────────────────
    image_files: list = []
    labels_list: list = []

    if os.path.isdir(data_path):
        # 1. Flat layout: NIfTI files alongside a companion labels.npy
        flat_labels_path = os.path.join(data_path, "labels.npy")
        if os.path.isfile(flat_labels_path):
            all_labels = np.load(flat_labels_path)
            nii_files  = sorted(
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith(".nii.gz") or f.endswith(".nii")
            )
            if len(nii_files) == len(all_labels):
                image_files = nii_files
                labels_list = all_labels.tolist()

        # 2. Class-subdirectory layout (fallback when flat layout yields nothing)
        if not image_files:
            for label_idx, subdir in enumerate(sorted(os.listdir(data_path))):
                subpath = os.path.join(data_path, subdir)
                if os.path.isdir(subpath):
                    for fname in sorted(os.listdir(subpath)):
                        if fname.endswith(".nii.gz") or fname.endswith(".nii"):
                            image_files.append(os.path.join(subpath, fname))
                            labels_list.append(label_idx)

    # ── Synthetic fallback ───────────────────────────────────────────────────
    if not image_files:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No NIfTI image files (.nii.gz / .nii) found under '{data_path}'. "
                "Provide a valid data_path containing real images, or set "
                "config['allow_synthetic_data'] = True to use random tensors for testing."
            )
        n           = config.get("synthetic_n", 40)
        num_classes = config.get("model_kwargs", {}).get("out_channels", 2)
        X           = torch.randn(n, 1, *spatial_size)
        y           = torch.randint(0, num_classes, (n,))
        full_ds     = TensorDataset(X, y)
        n_val       = max(1, int(n * val_ratio))
        n_train     = n - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
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

    # ── Real data: random_split then wrap with per-split MONAI transforms ────
    base_ds = _NiftiIndexDataset(image_files, np.array(labels_list, dtype=np.int64))
    n_val   = max(1, int(len(base_ds) * val_ratio))
    n_train = len(base_ds) - n_val
    train_subset, val_subset = random_split(
        base_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    if split == "train":
        ds      = _TransformSubset(train_subset, train_transforms)
        shuffle = True
    else:
        ds      = _TransformSubset(val_subset, val_transforms)
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
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
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
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
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs   = model(inputs)
    criterion = nn.CrossEntropyLoss()
    loss      = criterion(outputs, targets)
    return loss
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

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, TensorDataset

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


# ---------------------------------------------------------------------------
# Internal: wrap a random_split Subset with a split-specific MONAI transform.
#
# ImageDataset is instantiated WITHOUT a transform so that a single base
# dataset can be split first; this wrapper then applies the correct
# augmentation pipeline (train vs. val) per subset.
# ---------------------------------------------------------------------------

class _TransformSubset(torch.utils.data.Dataset):
    """Applies a MONAI-compatible transform to a random_split Subset.

    Because random_split shares the underlying dataset object between
    subsets, transforms cannot be embedded in ImageDataset itself if train
    and val pipelines differ.  This wrapper resolves that by deferring
    transform application to __getitem__ on a per-subset basis.
    """

    def __init__(self, subset: torch.utils.data.Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. build_model
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return a DenseNet121 for 3-D medical image classification.

    config keys consumed:
        model_kwargs (dict): forwarded to DenseNet121 constructor.
            Defaults: spatial_dims=3, in_channels=1, out_channels=2.
    """
    kwargs = config.get("model_kwargs", {})
    kwargs.setdefault("spatial_dims", 3)
    kwargs.setdefault("in_channels", 1)
    kwargs.setdefault("out_channels", 2)
    model = monai.networks.nets.DenseNet121(**kwargs)
    return model


# ---------------------------------------------------------------------------
# 2. build_dataloader
# ---------------------------------------------------------------------------

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    Real-data layout expected under config['data_path']:
        <data_path>/
            *.nii.gz          – NIfTI volumes (enumerated in sorted order)
            labels.npy        – int64 array, one entry per volume
                                (override with config['label_file'])

    The full dataset is split 80 / 20 (train / val) via random_split with a
    fixed seed.  Override the ratio with config['train_val_ratio'].

    Synthetic fallback:
        If real data is unavailable AND config['allow_synthetic_data'] is True,
        random tensors shaped (1, 96, 96, 96) are returned instead.
        If the flag is False (the default), FileNotFoundError is raised.

    config keys consumed:
        data_path           (str,   default ".")
        label_file          (str,   default "<data_path>/labels.npy")
        local.batch_size    (int,   default 16)
        train_val_ratio     (float, default 0.8)
        allow_synthetic_data (bool, default False)
        synthetic_n_samples (int,   default 20)
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    train_val_ratio = config.get("train_val_ratio", 0.8)

    train_transforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((96, 96, 96)),
        RandRotate90(),
    ])
    val_transforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((96, 96, 96)),
    ])

    # ------------------------------------------------------------------
    # Attempt to locate real data
    # ------------------------------------------------------------------
    image_files = []
    labels = None

    if os.path.isdir(data_path):
        image_files = sorted([
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.lower().endswith(".nii.gz") or f.lower().endswith(".nii")
        ])
        label_file = config.get(
            "label_file", os.path.join(data_path, "labels.npy")
        )
        if image_files and os.path.isfile(label_file):
            loaded = np.load(label_file).astype(np.int64)
            if len(loaded) == len(image_files):
                labels = loaded
            else:
                logging.warning(
                    "Label count (%d) != image count (%d) in '%s'; "
                    "ignoring labels file.",
                    len(loaded), len(image_files), label_file,
                )

    real_data_available = bool(image_files and labels is not None)

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------
    if not real_data_available:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real data not found at '{data_path}'. "
                "Expected NIfTI volumes (*.nii.gz) and a 'labels.npy' file "
                "containing one int64 label per volume. "
                "Set config['allow_synthetic_data'] = True to substitute "
                "synthetic random tensors for testing purposes only."
            )

        logging.warning(
            "Real data unavailable — using SYNTHETIC random data "
            "(allow_synthetic_data=True)."
        )
        n_samples = config.get("synthetic_n_samples", 20)
        n_train = max(1, int(n_samples * train_val_ratio))
        n_val = max(1, n_samples - n_train)

        x = torch.randn(n_samples, 1, 96, 96, 96)
        y = torch.randint(0, 2, (n_samples,), dtype=torch.long)
        full_ds = TensorDataset(x, y)

        train_ds, val_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen_ds = train_ds if split == "train" else val_ds
        return DataLoader(
            chosen_ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------
    # Real data path
    # ------------------------------------------------------------------
    # Build base dataset WITHOUT transforms so we can split first, then
    # apply split-specific transforms via _TransformSubset.
    base_ds = ImageDataset(image_files=image_files, labels=labels)

    total = len(base_ds)
    n_train = max(1, int(total * train_val_ratio))
    n_val = total - n_train
    if n_val < 1:
        n_val = 1
        n_train = total - n_val

    train_subset, val_subset = random_split(
        base_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    chosen_subset = train_subset if split == "train" else val_subset
    chosen_transform = train_transforms if split == "train" else val_transforms
    chosen_ds = _TransformSubset(chosen_subset, transform=chosen_transform)

    return DataLoader(
        chosen_ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# 3. train_step
# ---------------------------------------------------------------------------

def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with gradients attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); those calls must NOT appear here.

    Args:
        model:     The model returned by build_model(), already on its device.
        batch:     A (inputs, labels) tuple from the DataLoader.
        optimizer: The optimizer (passed by the FL runtime; unused here).
        config:    The run configuration dict (reserved for future use).

    Returns:
        loss (torch.Tensor): scalar CrossEntropyLoss with grad_fn intact.
    """
    device = _get_device(model)
    loss_fn = torch.nn.CrossEntropyLoss()

    inputs, labels = batch[0], batch[1]
    inputs = inputs.to(device, non_blocking=True)
    # Ensure labels are LongTensor as required by CrossEntropyLoss
    labels = labels.to(device, non_blocking=True).long()

    model.train()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    # Gradient remains attached — the FL runtime calls backward() externally.
    return loss
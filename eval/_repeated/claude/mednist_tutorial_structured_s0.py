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
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    TensorDataset,
    random_split,
)
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


# ── Dataset helper ──────────────────────────────────────────────────────────────

class _TransformSubset(Dataset):
    """
    Wraps a torch random_split Subset and applies a MONAI Compose transform
    to each image on retrieval.  This satisfies the MONAI-transform-in-Dataset
    requirement while allowing different augmentations for train vs. val.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ── FL API ──────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the 3-D DenseNet121 binary classifier.

    Extra constructor kwargs (e.g. spatial_dims, in_channels, out_channels)
    can be overridden through config['model_kwargs'].
    """
    kwargs = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
    }
    kwargs.update(config.get("model_kwargs", {}))
    return monai.networks.nets.DenseNet121(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ('train' or 'val').

    Config keys consumed:
        data_path              – directory that contains *.nii.gz files
                                 (default: ".")
        labels                 – list/array of integer labels aligned with the
                                 sorted list of discovered files (default: all 0)
        local.batch_size       – mini-batch size (default: 16)
        val_fraction           – fraction of data held out for validation
                                 (default: 0.2)
        seed                   – RNG seed for the split (default: 42)
        allow_synthetic_data   – if True and no real files are found, fall back
                                 to random tensors; if False, raise an error
        synthetic_n            – number of synthetic samples (default: 20)
    """
    batch_size   = config.get("local", {}).get("batch_size", 16)
    data_path    = config.get("data_path", ".")
    allow_synth  = config.get("allow_synthetic_data", False)
    val_fraction = config.get("val_fraction", 0.2)
    seed         = config.get("seed", 42)
    generator    = torch.Generator().manual_seed(seed)

    train_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()]
    )
    val_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))]
    )

    # ── Attempt to load real NIfTI data ────────────────────────────────────────
    data_dir    = Path(data_path)
    image_files = sorted(data_dir.glob("**/*.nii.gz")) if data_dir.is_dir() else []

    if image_files:
        image_files = [str(p) for p in image_files]
        n = len(image_files)

        provided = config.get("labels", None)
        if provided is not None:
            labels = np.array(provided, dtype=np.int64)
        else:
            labels = np.zeros(n, dtype=np.int64)
            logging.warning(
                "No 'labels' key found in config; defaulting to all-zero labels. "
                "Supply config['labels'] for meaningful training."
            )

        # Build a single base dataset (no transforms yet) then split it
        base_ds = ImageDataset(image_files=image_files, labels=labels)
        n_val   = max(1, round(n * val_fraction))
        n_train = n - n_val
        train_subset, val_subset = random_split(base_ds, [n_train, n_val],
                                                generator=generator)

        chosen_subset    = train_subset if split == "train" else val_subset
        chosen_transform = train_transforms if split == "train" else val_transforms
        dataset          = _TransformSubset(chosen_subset, chosen_transform)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    # ── No real data found ─────────────────────────────────────────────────────
    if not allow_synth:
        raise FileNotFoundError(
            f"No .nii.gz files found under '{data_path}'. "
            "Set config['allow_synthetic_data'] = True to use synthetic tensors "
            "for smoke-testing the FL pipeline."
        )

    logging.warning(
        "allow_synthetic_data=True: training on randomly generated volumes. "
        "This should only be used for pipeline smoke-tests."
    )
    n_synth  = config.get("synthetic_n", 20)
    images_t = torch.randn(n_synth, 1, 96, 96, 96)
    labels_t = torch.randint(0, 2, (n_synth,))
    full_ds  = TensorDataset(images_t, labels_t)

    n_val   = max(1, round(n_synth * val_fraction))
    n_train = n_synth - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    chosen = train_ds if split == "train" else val_ds
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        pin_memory=torch.cuda.is_available(),
    )


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Execute one forward pass and return the scalar loss WITH grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.
    """
    device = next(model.parameters()).device

    inputs = batch[0].to(device)
    labels = batch[1].to(device).long()   # CrossEntropyLoss requires Long

    outputs  = model(inputs)
    loss_fn  = torch.nn.CrossEntropyLoss()
    loss     = loss_fn(outputs, labels)
    return loss                            # grad graph kept intact – no .backward()
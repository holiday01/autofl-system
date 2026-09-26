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
from torch.utils.data import TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


class _NiftiDataset(torch.utils.data.Dataset):
    """Wraps MONAI ImageDataset, applying MONAI transforms inside __getitem__.

    Keeping transforms here (rather than passed to ImageDataset directly) lets
    random_split share one underlying file-list while still allowing the caller
    to swap transforms per split.
    """

    def __init__(self, image_files, labels, transform):
        self._image_ds = ImageDataset(image_files=image_files, labels=labels)
        self._transform = transform

    def __len__(self):
        return len(self._image_ds)

    def __getitem__(self, idx):
        img, label = self._image_ds[idx]
        img = self._transform(img)
        return img, label


# ---------------------------------------------------------------------------
# FL contract
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the 3-D DenseNet121 classifier.

    Supported model_kwargs keys (all optional, defaults match the original
    script):
        spatial_dims  (int, default 3)
        in_channels   (int, default 1)
        out_channels  (int, default 2)
    """
    kw = config.get("model_kwargs", {})
    return monai.networks.nets.DenseNet121(
        spatial_dims=kw.get("spatial_dims", 3),
        in_channels=kw.get("in_channels", 1),
        out_channels=kw.get("out_channels", 2),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    Config keys read:
        config["local"]["batch_size"]        – default 16
        config["data_path"]                  – root directory of NIfTI files
        config["allow_synthetic_data"]       – must be True to enable fallback
        config["local"]["synthetic_samples"] – number of fake volumes (default 20)

    Real-data layout expected under data_path:
        *.nii.gz / *.nii   – one file per sample, sorted alphabetically
        labels.npy          – 1-D int64 array aligned with the sorted file list
          OR
        labels.csv          – two-column CSV; column index 1 holds the int label

    Transforms:
        train: ScaleIntensity → EnsureChannelFirst → Resize(96³) → RandRotate90
        val:   ScaleIntensity → EnsureChannelFirst → Resize(96³)

    random_split is used to derive the 80/20 train/val partition; the val
    subset is then rebuilt with augmentation-free val_transforms.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    train_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()]
    )
    val_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))]
    )

    # ── Discover real data ──────────────────────────────────────────────────
    image_files: list = []
    labels = None

    if os.path.isdir(data_path):
        candidates = sorted(
            f for f in os.listdir(data_path)
            if f.endswith(".nii.gz") or f.endswith(".nii")
        )
        if candidates:
            image_files = [os.path.join(data_path, f) for f in candidates]
            for labels_fname in ("labels.npy", "labels.csv"):
                labels_path = os.path.join(data_path, labels_fname)
                if os.path.exists(labels_path):
                    if labels_fname.endswith(".npy"):
                        labels = np.load(labels_path).astype(np.int64)
                    else:
                        labels = np.loadtxt(
                            labels_path, delimiter=",", dtype=np.int64, usecols=1
                        )
                    break
            if labels is None:
                raise FileNotFoundError(
                    f"Found {len(image_files)} NIfTI file(s) in '{data_path}' but no "
                    "labels file. Provide 'labels.npy' (1-D int64 array aligned with "
                    "the sorted file list) or 'labels.csv' (col index 1 = int label)."
                )

    # ── Synthetic fallback ──────────────────────────────────────────────────
    if not image_files:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No NIfTI image files found at data_path='{data_path}'. "
                "Provide a valid data_path containing .nii/.nii.gz files, "
                "or set config['allow_synthetic_data']=True to use randomly "
                "generated tensors for smoke-testing only."
            )
        n_synth = config.get("local", {}).get("synthetic_samples", 20)
        X = torch.randn(n_synth, 1, 96, 96, 96)
        y = torch.randint(0, 2, (n_synth,), dtype=torch.long)
        full_ds = TensorDataset(X, y)
        n_train = max(1, int(n_synth * 0.8))
        n_val = max(1, n_synth - n_train)
        train_sub, val_sub = random_split(full_ds, [n_train, n_val])
        subset = train_sub if split == "train" else val_sub
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    # ── Real data ───────────────────────────────────────────────────────────
    # Build a single full dataset (with train_transforms) and use random_split
    # to determine the partition, then rebuild the val subset with val_transforms
    # so that no random augmentation leaks into evaluation.
    full_ds = _NiftiDataset(image_files, labels, train_transforms)
    n_total = len(full_ds)
    n_train = max(1, int(n_total * 0.8))
    n_val = max(1, n_total - n_train)
    train_sub, val_sub = random_split(full_ds, [n_train, n_val])

    if split == "train":
        return DataLoader(
            train_sub,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    # Rebuild the val partition with augmentation-free val_transforms.
    val_indices = list(val_sub.indices)
    val_ds = _NiftiDataset(
        [image_files[i] for i in val_indices],
        labels[val_indices],
        val_transforms,
    )
    return DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the cross-entropy loss with grad attached.

    The FL runtime is responsible for loss.backward() and optimizer.step().
    """
    device = next(model.parameters()).device
    inputs, targets = batch[0].to(device), batch[1].to(device)
    outputs = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(outputs, targets)
    return loss
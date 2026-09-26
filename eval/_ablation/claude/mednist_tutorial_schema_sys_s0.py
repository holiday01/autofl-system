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
from torch.utils.data import Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


# ---------------------------------------------------------------------------
# IXI-T1 file list and gender labels from the original script
# ---------------------------------------------------------------------------
_IMAGE_FILENAMES = [
    "IXI314-IOP-0889-T1.nii.gz",
    "IXI249-Guys-1072-T1.nii.gz",
    "IXI609-HH-2600-T1.nii.gz",
    "IXI173-HH-1590-T1.nii.gz",
    "IXI020-Guys-0700-T1.nii.gz",
    "IXI342-Guys-0909-T1.nii.gz",
    "IXI134-Guys-0780-T1.nii.gz",
    "IXI577-HH-2661-T1.nii.gz",
    "IXI066-Guys-0731-T1.nii.gz",
    "IXI130-HH-1528-T1.nii.gz",
    "IXI607-Guys-1097-T1.nii.gz",
    "IXI175-HH-1570-T1.nii.gz",
    "IXI385-HH-2078-T1.nii.gz",
    "IXI344-Guys-0905-T1.nii.gz",
    "IXI409-Guys-0960-T1.nii.gz",
    "IXI584-Guys-1129-T1.nii.gz",
    "IXI253-HH-1694-T1.nii.gz",
    "IXI092-HH-1436-T1.nii.gz",
    "IXI574-IOP-1156-T1.nii.gz",
    "IXI585-Guys-1130-T1.nii.gz",
]

_LABELS = np.array(
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    dtype=np.int64,
)


# ---------------------------------------------------------------------------
# Dataset helpers — MONAI transforms are wrapped inside Dataset classes
# per FL conversion rules.
# ---------------------------------------------------------------------------

class _IXIDataset(Dataset):
    """
    Thin wrapper around MONAI ImageDataset that owns the MONAI Compose
    transform pipeline.  Allows random_split to operate on a single object
    while still letting each split carry its own transform (train vs val).
    """

    def __init__(self, image_files, labels, transform):
        self._inner = ImageDataset(
            image_files=image_files,
            labels=labels,
            transform=transform,
        )

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]


class _SyntheticMRIDataset(Dataset):
    """
    Synthetic 3-D MRI dataset used only when
    config['allow_synthetic_data'] is True and real data is absent.
    Produces random (1, 96, 96, 96) float tensors and random binary labels.
    """

    def __init__(self, size: int = 20, spatial_shape=(96, 96, 96), num_classes: int = 2):
        self.size = size
        self.spatial_shape = spatial_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(1, *self.spatial_shape)
        label = torch.randint(0, self.num_classes, ()).long()
        return image, label


# ---------------------------------------------------------------------------
# Lightweight index proxy — lets random_split determine the fold split
# without loading any actual data.
# ---------------------------------------------------------------------------

class _IndexProxy(Dataset):
    """Returns its own index; used solely to obtain shuffled split indices."""

    def __init__(self, n: int):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return i


# ---------------------------------------------------------------------------
# FL interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate MONAI DenseNet121 for 3-D binary (gender) classification.

    Supported model_kwargs:
        spatial_dims  (int, default 3)
        in_channels   (int, default 1)
        out_channels  (int, default 2)
    """
    kwargs = config.get("model_kwargs", {})
    model = monai.networks.nets.DenseNet121(
        spatial_dims=kwargs.get("spatial_dims", 3),
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 2),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the IXI-T1 dataset.

    The full dataset is split 80 / 20 (train / val) using random_split with
    a fixed seed for reproducibility.  Train receives augmentation
    (RandRotate90); val does not.

    If the data is missing and config['allow_synthetic_data'] is True, a
    synthetic fallback is used.  If the flag is False (the default), a
    FileNotFoundError is raised instead.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got '{split!r}'")

    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    images = [os.path.join(data_path, f) for f in _IMAGE_FILENAMES]
    data_available = all(os.path.exists(p) for p in images)

    # ------------------------------------------------------------------ #
    # Synthetic fallback (gated — never silently active)                  #
    # ------------------------------------------------------------------ #
    if not data_available:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"IXI-T1 dataset not found under '{data_path}'. "
                "Provide the real data or set config['allow_synthetic_data'] = True "
                "to use synthetic tensors for debugging."
            )
        full_ds = _SyntheticMRIDataset(size=20)
        n_val = max(1, int(0.2 * len(full_ds)))
        n_train = len(full_ds) - n_val
        train_sub, val_sub = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_sub if split == "train" else val_sub
        return DataLoader(
            chosen,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------ #
    # Real data path                                                       #
    # ------------------------------------------------------------------ #
    train_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()]
    )
    val_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))]
    )

    # Use random_split on a lightweight proxy to obtain reproducible split
    # indices, then materialise two ImageDataset instances — one per split —
    # each carrying the correct MONAI transform pipeline.
    n_total = len(images)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    train_proxy, val_proxy = random_split(
        _IndexProxy(n_total),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_indices = list(train_proxy.indices)
    val_indices = list(val_proxy.indices)

    if split == "train":
        subset_images = [images[i] for i in train_indices]
        subset_labels = _LABELS[train_indices]
        dataset = _IXIDataset(subset_images, subset_labels, train_transforms)
        shuffle = True
    else:
        subset_images = [images[i] for i in val_indices]
        subset_labels = _LABELS[val_indices]
        dataset = _IXIDataset(subset_images, subset_labels, val_transforms)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Execute a single forward pass and return the loss tensor with grad
    attached.  The FL runtime is responsible for calling loss.backward()
    and optimizer.step(); neither is called here.
    """
    device = next(model.parameters()).device
    inputs = batch[0].to(device)
    labels = batch[1].to(device)

    outputs = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    return loss
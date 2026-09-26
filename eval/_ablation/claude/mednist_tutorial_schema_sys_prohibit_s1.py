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


# ── Dataset helpers (MONAI rule: wrap MONAI transforms inside the Dataset class) ──

class _IXIDataset(Dataset):
    """Full IXI-T1 dataset with MONAI transforms baked in."""

    def __init__(self, image_files: list, labels: np.ndarray, transform: Compose):
        self.inner = ImageDataset(
            image_files=image_files,
            labels=labels,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx):
        return self.inner[idx]


class _AugWrapper(Dataset):
    """Applies an extra per-sample MONAI transform on top of an existing Subset.

    Used to add RandRotate90 only to the training split after random_split,
    without re-creating the underlying ImageDataset.
    """

    def __init__(self, subset, aug_transform):
        self.subset = subset
        self.aug = aug_transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.aug(img), label


class _SyntheticDataset(Dataset):
    """Synthetic 3-D image/label pairs for offline / CI runs.

    Only instantiated when config['allow_synthetic_data'] is explicitly True.
    Images have the same shape as processed IXI-T1 volumes: (1, 96, 96, 96).
    Labels are random binary integers.
    """

    def __init__(self, length: int = 20, spatial_size: tuple = (96, 96, 96)):
        self.length = length
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        img = torch.randn(1, *self.spatial_size)
        label = torch.randint(0, 2, ()).long()
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# FL API
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate MONAI DenseNet121 for 3-D binary classification.

    config keys consumed:
        model_kwargs (dict): forwarded verbatim to DenseNet121.__init__.
            Defaults: spatial_dims=3, in_channels=1, out_channels=2.
    """
    kwargs = dict(config.get("model_kwargs", {}))
    kwargs.setdefault("spatial_dims", 3)
    kwargs.setdefault("in_channels", 1)
    kwargs.setdefault("out_channels", 2)
    return monai.networks.nets.DenseNet121(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    config keys consumed:
        local.batch_size  (int, default 16)
        data_path         (str, default ./workspace/data/medical/ixi/IXI-T1)
        allow_synthetic_data (bool, default False)

    Strategy
    --------
    1. A single _IXIDataset is built with deterministic base transforms
       (ScaleIntensity → EnsureChannelFirst → Resize).
    2. random_split(80 / 20) partitions it reproducibly.
    3. The training subset is wrapped in _AugWrapper to add RandRotate90,
       matching the original script's train vs. val transform difference.
    4. If real data is absent and allow_synthetic_data is False, a
       FileNotFoundError is raised immediately — no silent fallback.
    """
    batch_size: int = config.get("local", {}).get("batch_size", 16)
    data_path: str = config.get(
        "data_path",
        os.sep.join([".", "workspace", "data", "medical", "ixi", "IXI-T1"]),
    )

    _IMAGE_NAMES = [
        "IXI314-IOP-0889-T1.nii.gz", "IXI249-Guys-1072-T1.nii.gz",
        "IXI609-HH-2600-T1.nii.gz",  "IXI173-HH-1590-T1.nii.gz",
        "IXI020-Guys-0700-T1.nii.gz", "IXI342-Guys-0909-T1.nii.gz",
        "IXI134-Guys-0780-T1.nii.gz", "IXI577-HH-2661-T1.nii.gz",
        "IXI066-Guys-0731-T1.nii.gz", "IXI130-HH-1528-T1.nii.gz",
        "IXI607-Guys-1097-T1.nii.gz", "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz",  "IXI344-Guys-0905-T1.nii.gz",
        "IXI409-Guys-0960-T1.nii.gz", "IXI584-Guys-1129-T1.nii.gz",
        "IXI253-HH-1694-T1.nii.gz",  "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz",  "IXI585-Guys-1130-T1.nii.gz",
    ]
    _LABELS = np.array(
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        dtype=np.int64,
    )

    image_files = [os.path.join(data_path, f) for f in _IMAGE_NAMES]
    data_available = os.path.isdir(data_path) and all(
        os.path.isfile(p) for p in image_files
    )

    # ── Synthetic-data path ────────────────────────────────────────────────
    if not data_available:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"IXI-T1 NIfTI files not found under '{data_path}'. "
                "Download the IXI-T1 dataset or set "
                "config['allow_synthetic_data'] = True to use synthetic "
                "random tensors for offline / CI testing."
            )
        full_ds = _SyntheticDataset(length=20)
        n_train, n_val = 16, 4
        train_subset, val_subset = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_subset if split == "train" else val_subset
        return DataLoader(
            chosen,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    # ── Real-data path ─────────────────────────────────────────────────────
    # Build one dataset with deterministic (val-equivalent) transforms so that
    # random_split operates on a single, consistent object.
    base_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))]
    )
    full_ds = _IXIDataset(
        image_files=image_files,
        labels=_LABELS,
        transform=base_transforms,
    )

    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_subset, val_subset = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Re-apply RandRotate90 only for the training subset, mirroring the
    # original script's train_transforms vs val_transforms distinction.
    if split == "train":
        chosen = _AugWrapper(train_subset, RandRotate90())
    else:
        chosen = val_subset

    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Single forward pass for one FL training round.

    Responsibilities of the FL runtime (NOT done here):
        • loss.backward()
        • optimizer.step() / optimizer.zero_grad()

    Returns the live loss tensor (grad_fn intact).
    """
    device = next(model.parameters()).device
    inputs = batch[0].to(device)
    labels = batch[1].to(device)

    outputs = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    return loss  # grad_fn preserved; do NOT detach / call .item()
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
from torch.utils.data import DataLoader, Dataset, random_split

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


# ── Synthetic fallback dataset ───────────────────────────────────────────────

class _SyntheticMRIDataset(Dataset):
    """
    Generates random 3-D brain-MRI-shaped tensors and binary labels.
    Used as a drop-in fallback when the real IXI-T1 files are unavailable.
    MONAI transforms are baked into __getitem__ so the tensors are already
    channel-first and spatially resized, matching what ImageDataset produces.
    """

    def __init__(self, length: int = 20, spatial_size: tuple = (96, 96, 96)):
        self.length = length
        self.spatial_size = spatial_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        # Shape: (1, D, H, W) — channel-first, matching EnsureChannelFirst + Resize
        image = torch.randn(1, *self.spatial_size)
        label = torch.randint(0, 2, (1,)).squeeze().long()
        return image, label


# ── IXI-T1 metadata (mirrors the original script) ───────────────────────────

_IXI_FILENAMES = [
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

_IXI_LABELS = np.array(
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    dtype=np.int64,
)

# MONAI transforms — wrapped inside ImageDataset (per original script pattern)
_TRAIN_TRANSFORMS = Compose(
    [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()]
)
_VAL_TRANSFORMS = Compose(
    [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))]
)


# ── FL client interface ──────────────────────────────────────────────────────

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return a DenseNet121 for 3-D medical image classification.

    Defaults match the original script (spatial_dims=3, in_channels=1,
    out_channels=2). Override any kwarg via config["model_kwargs"].
    """
    kwargs = config.get("model_kwargs", {})
    kwargs.setdefault("spatial_dims", 3)
    kwargs.setdefault("in_channels", 1)
    kwargs.setdefault("out_channels", 2)
    return monai.networks.nets.DenseNet121(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Real data path is read from config["data_path"]; batch size from
    config["local"]["batch_size"] (default 16).  A deterministic
    80/20 random_split (seed 42) is applied to the full dataset so
    both splits are non-overlapping.  Falls back to _SyntheticMRIDataset
    when the IXI-T1 .nii.gz files are not present on disk.
    """
    batch_size: int = config.get("local", {}).get("batch_size", 16)
    data_path: str = config.get(
        "data_path",
        os.path.join(".", "workspace", "data", "medical", "ixi", "IXI-T1"),
    )

    image_files = [os.path.join(data_path, f) for f in _IXI_FILENAMES]
    real_available = all(os.path.isfile(p) for p in image_files)

    _SPLIT_SEED = 42

    if real_available:
        # Build two full ImageDatasets — one per transform set — then carve out
        # the same indices via random_split so the split boundary is consistent.
        transforms = _TRAIN_TRANSFORMS if split == "train" else _VAL_TRANSFORMS
        full_ds = ImageDataset(
            image_files=image_files,
            labels=_IXI_LABELS,
            transform=transforms,
        )
        n_total = len(full_ds)
        n_val = max(1, int(n_total * 0.2))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(_SPLIT_SEED)
        train_subset, val_subset = random_split(full_ds, [n_train, n_val],
                                                generator=generator)
        dataset = train_subset if split == "train" else val_subset
    else:
        logging.warning(
            "IXI-T1 data not found at '%s'. Using synthetic fallback dataset.",
            data_path,
        )
        full_ds = _SyntheticMRIDataset(length=20)
        n_total = len(full_ds)
        n_val = max(1, int(n_total * 0.2))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(_SPLIT_SEED)
        train_subset, val_subset = random_split(full_ds, [n_train, n_val],
                                                generator=generator)
        dataset = train_subset if split == "train" else val_subset

    return DataLoader(
        dataset,
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
    """
    Execute a single forward pass and return the scalar loss (grad attached).

    The FL runtime is responsible for loss.backward() and optimizer.step();
    this function must NOT call either.
    """
    device = next(model.parameters()).device

    inputs: torch.Tensor = batch[0].to(device)
    labels: torch.Tensor = batch[1].to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    outputs = model(inputs)          # (N, 2) logits
    loss = loss_fn(outputs, labels)  # scalar, grad_fn retained
    return loss
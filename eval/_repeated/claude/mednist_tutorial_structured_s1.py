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

import glob
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


# ---------------------------------------------------------------------------
# IXI-T1 manifest preserved from the original script
# ---------------------------------------------------------------------------
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
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64
)


# ---------------------------------------------------------------------------
# Helper: wraps a random_split Subset and applies the correct MONAI transform
# ---------------------------------------------------------------------------
class _MONAITransformSubset(Dataset):
    """Applies a MONAI Compose transform on top of a random_split Subset.

    The parent ImageDataset is built with transform=None so that raw arrays
    flow through random_split unchanged; this wrapper then applies the
    split-appropriate pipeline (augmentation for train, deterministic for val).
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


# ---------------------------------------------------------------------------
# Helper: fully synthetic dataset (3-D volumes + random binary labels)
# ---------------------------------------------------------------------------
class _SyntheticMedicalDataset(Dataset):
    """Synthetic stand-in for IXI T1 volumes.  Gated by allow_synthetic_data."""

    def __init__(self, num_samples: int = 20, spatial_size=(96, 96, 96), num_classes: int = 2):
        self.num_samples = num_samples
        self.spatial_size = spatial_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(1, *self.spatial_size)          # (C, D, H, W)
        label = torch.randint(0, self.num_classes, ()).long()
        return image, label


# ===========================================================================
# FL API
# ===========================================================================

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate MONAI DenseNet121 for 3-D medical image classification.

    Supported model_kwargs keys (all optional):
        spatial_dims  – default 3
        in_channels   – default 1
        out_channels  – default 2
    """
    kwargs = config.get("model_kwargs", {})
    model = monai.networks.nets.DenseNet121(
        spatial_dims=int(kwargs.get("spatial_dims", 3)),
        in_channels=int(kwargs.get("in_channels", 1)),
        out_channels=int(kwargs.get("out_channels", 2)),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for *split* ("train" or "val").

    Data discovery order
    --------------------
    1.  Look for the 20 IXI-T1 filenames under ``config["data_path"]``.
    2.  Fall back to a recursive glob for any ``*.nii.gz`` in that directory
        (labels default to 0).
    3.  If no files are found AND ``config["allow_synthetic_data"]`` is True,
        return a DataLoader backed by ``_SyntheticMedicalDataset``.
    4.  If no files are found AND the flag is False (or absent), raise
        ``FileNotFoundError`` – never silently train on synthetic data.

    The train/val split is produced by ``torch.utils.data.random_split`` on a
    single ``ImageDataset`` (built without transforms), with an 80/20 ratio.
    Split-appropriate MONAI transforms are then applied via
    ``_MONAITransformSubset``.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path  = config.get("data_path", ".")

    # MONAI transform pipelines (identical to the original script)
    train_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()]
    )
    val_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))]
    )

    # ------------------------------------------------------------------
    # 1. Try the known IXI manifest
    # ------------------------------------------------------------------
    candidate_paths = [os.path.join(data_path, f) for f in _IXI_FILENAMES]
    available_paths = [p for p in candidate_paths if os.path.isfile(p)]

    if available_paths:
        label_map   = dict(zip(candidate_paths, _IXI_LABELS))
        file_labels = np.array([label_map[p] for p in available_paths], dtype=np.int64)
    else:
        # ------------------------------------------------------------------
        # 2. Scan directory for any NIfTI files
        # ------------------------------------------------------------------
        scanned = sorted(glob.glob(os.path.join(data_path, "**", "*.nii.gz"), recursive=True))
        if scanned:
            available_paths = scanned
            file_labels     = np.zeros(len(scanned), dtype=np.int64)
        else:
            # ------------------------------------------------------------------
            # 3/4. No real data found
            # ------------------------------------------------------------------
            if not config.get("allow_synthetic_data", False):
                raise FileNotFoundError(
                    f"No NIfTI (.nii.gz) files found at data_path='{data_path}'. "
                    "Supply real IXI T1 data or set config['allow_synthetic_data']=True "
                    "to enable synthetic volume generation for debugging."
                )

            # Synthetic path (gated above)
            num_samples = int(config.get("synthetic_num_samples", 20))
            full_synthetic = _SyntheticMedicalDataset(num_samples=num_samples)
            n_train = max(1, int(num_samples * 0.8))
            n_val   = max(1, num_samples - n_train)
            if n_train + n_val != num_samples:
                n_val = num_samples - n_train
            train_sub, val_sub = random_split(full_synthetic, [n_train, n_val])
            chosen  = train_sub if split == "train" else val_sub
            return DataLoader(
                chosen,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )

    # ------------------------------------------------------------------
    # Build a single ImageDataset (no transforms) then random_split it
    # ------------------------------------------------------------------
    base_ds = ImageDataset(
        image_files=available_paths,
        labels=file_labels,
        transform=None,           # transforms applied per-split below
    )

    n_total = len(base_ds)
    n_train = max(1, int(n_total * 0.8))
    n_val   = max(1, n_total - n_train)
    if n_train + n_val != n_total:
        n_val = n_total - n_train

    train_subset, val_subset = random_split(base_ds, [n_train, n_val])

    if split == "train":
        ds      = _MONAITransformSubset(train_subset, train_transforms)
        shuffle = True
    else:
        ds      = _MONAITransformSubset(val_subset, val_transforms)
        shuffle = False

    return DataLoader(
        ds,
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
    """Run a single forward pass and return the loss tensor (grad attached).

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function intentionally does neither.
    """
    device = next(model.parameters()).device

    inputs = batch[0].to(device)
    labels = batch[1].to(device).long()   # CrossEntropyLoss requires LongTensor

    loss_function = torch.nn.CrossEntropyLoss()
    outputs = model(inputs)
    loss    = loss_function(outputs, labels)

    # loss.backward() and optimizer.step() are intentionally omitted –
    # the FL runtime handles the full gradient update cycle.
    return loss
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
from torch.utils.data import DataLoader, TensorDataset, random_split

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _NiftiDataset(torch.utils.data.Dataset):
    """Raw NIfTI loader with no transforms applied.

    Keeping loading and transforms separate lets random_split operate on
    the full corpus first; per-split transforms are applied afterwards by
    _TransformDataset, so augmentation never leaks into the val split.
    """

    def __init__(self, image_files, labels):
        self.image_files = image_files
        self.labels = labels
        self._load = monai.transforms.LoadImage(image_only=True, ensure_channel_first=False)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = self._load(self.image_files[idx])
        return img, self.labels[idx]


class _TransformDataset(torch.utils.data.Dataset):
    """Wraps a torch.utils.data.Subset and applies a MONAI Compose lazily."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate MONAI DenseNet121 for 3-D binary classification.

    Supported model_kwargs keys:
        spatial_dims  (int, default 3)
        in_channels   (int, default 1)
        out_channels  (int, default 2)
    """
    kw = config.get("model_kwargs", {})
    model = monai.networks.nets.DenseNet121(
        spatial_dims=kw.get("spatial_dims", 3),
        in_channels=kw.get("in_channels", 1),
        out_channels=kw.get("out_channels", 2),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for *split* ('train' or 'val').

    Data discovery
    --------------
    Scans config['data_path'] for ``*.nii.gz`` files and a ``labels.npy``
    array (shape ``[N]``, dtype int64).  A single _NiftiDataset is built from
    the full corpus and split with torch.utils.data.random_split so that the
    train/val partition is consistent across calls.

    Synthetic fallback
    ------------------
    If real data cannot be found **and** ``config['allow_synthetic_data']``
    is ``True``, random tensors are generated for smoke-testing.  If the flag
    is absent or ``False`` a ``FileNotFoundError`` is raised instead — synthetic
    data is never injected silently.

    Config keys consumed
    --------------------
    data_path              str   root directory that holds *.nii.gz + labels.npy
    local.batch_size       int   mini-batch size (default 16)
    train_ratio            float fraction assigned to training (default 0.8)
    seed                   int   optional RNG seed for the random_split
    allow_synthetic_data   bool  enable synthetic fallback (default False)
    synthetic_samples      int   number of synthetic items (default 20)
    model_kwargs.in_channels   int  used by synthetic generator (default 1)
    model_kwargs.out_channels  int  number of classes for synthetic labels (default 2)
    model_kwargs.spatial_size  list synthetic volume shape (default [96,96,96])
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path  = config.get("data_path", ".")
    train_ratio = config.get("train_ratio", 0.8)
    seed = config.get("seed", None)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # Transforms — identical to the original script
    train_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()]
    )
    val_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))]
    )

    # ------------------------------------------------------------------
    # Attempt to load real NIfTI data
    # ------------------------------------------------------------------
    image_files = sorted(glob.glob(os.path.join(data_path, "*.nii.gz")))
    labels_path = os.path.join(data_path, "labels.npy")

    if image_files and os.path.isfile(labels_path):
        labels = np.load(labels_path).astype(np.int64)

        if len(labels) != len(image_files):
            raise ValueError(
                f"Mismatch: found {len(image_files)} image files but "
                f"{len(labels)} entries in '{labels_path}'."
            )

        raw_ds = _NiftiDataset(image_files, labels)
        n = len(raw_ds)
        n_train = max(1, int(n * train_ratio))
        n_val = n - n_train          # absorbs any rounding remainder

        train_sub, val_sub = random_split(raw_ds, [n_train, n_val], generator=generator)

        ds = _TransformDataset(
            train_sub if split == "train" else val_sub,
            train_transforms if split == "train" else val_transforms,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------
    # Real data unavailable — honour the synthetic-data gate
    # ------------------------------------------------------------------
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"No '*.nii.gz' image files or 'labels.npy' label array found "
            f"under '{data_path}'.  Supply real IXI-style NIfTI data, or set "
            "config['allow_synthetic_data'] = True to enable the synthetic "
            "tensor fallback (for smoke-testing / CI only — never for real FL)."
        )

    # Synthetic fallback — random tensors shaped like real volumes
    kw = config.get("model_kwargs", {})
    in_channels  = kw.get("in_channels", 1)
    out_channels = kw.get("out_channels", 2)
    spatial_size = tuple(kw.get("spatial_size", [96, 96, 96]))
    n_synthetic  = config.get("synthetic_samples", 20)

    syn_images = torch.randn(n_synthetic, in_channels, *spatial_size)
    syn_labels = torch.randint(0, out_channels, (n_synthetic,), dtype=torch.long)

    full_ds = TensorDataset(syn_images, syn_labels)
    n_train = max(1, int(n_synthetic * train_ratio))
    n_val   = n_synthetic - n_train

    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)
    ds = train_ds if split == "train" else val_ds
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Execute one forward pass and return the live loss tensor.

    The FL runtime owns the backward pass and the parameter update; this
    function must NOT call loss.backward() or optimizer.step().
    The returned tensor retains its grad_fn so the runtime can differentiate
    through it.
    """
    loss_fn = torch.nn.CrossEntropyLoss()

    device = next(model.parameters()).device
    inputs = batch[0].to(device)
    labels = batch[1].to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    # Return the live tensor — do NOT call .detach() or .item()
    return loss
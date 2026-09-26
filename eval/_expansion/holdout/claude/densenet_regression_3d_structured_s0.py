import logging
import os
import sys
import shutil
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import numpy as np

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from monai.networks.nets import Regressor

pin_memory = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# FL API — build_model
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the MONAI Regressor.

    Supported model_kwargs keys (all optional, defaults match the original
    notebook):
        in_shape  : list/tuple  – default [1, 96, 96, 96]
        out_shape : int         – default 1
        channels  : list/tuple  – default (16, 32, 64, 128, 256)
        strides   : list/tuple  – default (2, 2, 2, 2)
    """
    kw = config.get("model_kwargs", {})
    model = Regressor(
        in_shape=kw.get("in_shape", [1, 96, 96, 96]),
        out_shape=kw.get("out_shape", 1),
        channels=tuple(kw.get("channels", (16, 32, 64, 128, 256))),
        strides=tuple(kw.get("strides", (2, 2, 2, 2))),
    )
    return model


# ---------------------------------------------------------------------------
# Helper – synthetic dataset (used only when allow_synthetic_data=True)
# ---------------------------------------------------------------------------

class _SyntheticIXIDataset(Dataset):
    """Random-tensor stand-in for the IXI brain MRI + age dataset."""

    def __init__(self, size: int = 20, input_shape: tuple = (1, 96, 96, 96)):
        self.size = size
        self.input_shape = input_shape

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        image = torch.randn(*self.input_shape)
        label = torch.randint(20, 90, (1,)).float().squeeze()
        return image, label


# ---------------------------------------------------------------------------
# FL API — build_dataloader
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

_IXI_AGES = np.array([
    45.86, 68.27, 29.00, 29.57, 39.47,
    48.68, 47.35, 64.19, 46.17, 38.77,
    83.81, 72.27, 64.65, 62.09, 70.95,
    41.33, 24.00, 33.24, 50.57, 28.12,
])


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    Config keys consumed:
        data_path               – directory that contains an ``ixi/`` sub-folder
                                  (default ".")
        local.batch_size        – mini-batch size (default 16)
        allow_synthetic_data    – if True and real files are absent, fall back
                                  to synthetic random tensors; if False (default)
                                  and files are absent, raise FileNotFoundError.

    Train/val partitioning is done with torch.utils.data.random_split
    (seed 42, 80 / 20 split) so every call with the same config is
    reproducible across clients.
    """
    local_cfg = config.get("local", {})
    batch_size = local_cfg.get("batch_size", 16)
    data_path = config.get("data_path", ".")

    # ------------------------------------------------------------------
    # MONAI transforms (preserved exactly from the original notebook)
    # ------------------------------------------------------------------
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

    ixi_dir = os.path.join(data_path, "ixi")
    images = [os.path.join(ixi_dir, fn) for fn in _IXI_FILENAMES]
    data_available = all(os.path.isfile(p) for p in images)

    # ------------------------------------------------------------------
    # Branch: real data unavailable
    # ------------------------------------------------------------------
    if not data_available:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"IXI NIfTI files not found under '{ixi_dir}'. "
                "Point config['data_path'] at the directory that contains the "
                "'ixi/' sub-folder, or set config['allow_synthetic_data'] = True "
                "to fall back to synthetic random tensors."
            )

        # Synthetic fallback — gated on allow_synthetic_data above
        n_total = 20
        n_train = 16
        n_val = n_total - n_train
        full_ds = _SyntheticIXIDataset(size=n_total, input_shape=(1, 96, 96, 96))
        train_ds, val_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        subset = train_ds if split == "train" else val_ds
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=pin_memory,
        )

    # ------------------------------------------------------------------
    # Branch: real data available
    # Use random_split on a thin index-only TensorDataset so the
    # partition is reproducible while each split gets its own MONAI
    # transform pipeline.
    # ------------------------------------------------------------------
    n_total = len(images)
    n_train = max(1, int(0.8 * n_total))
    n_val = n_total - n_train

    _index_ds = TensorDataset(torch.arange(n_total))
    train_subset, val_subset = random_split(
        _index_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_indices = list(train_subset.indices)
    val_indices = list(val_subset.indices)

    if split == "train":
        sel_images = [images[i] for i in train_indices]
        sel_ages = _IXI_AGES[train_indices]
        xfm = train_transforms
    else:
        sel_images = [images[i] for i in val_indices]
        sel_ages = _IXI_AGES[val_indices]
        xfm = val_transforms

    dataset = ImageDataset(
        image_files=sel_images,
        labels=sel_ages,
        transform=xfm,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# FL API — train_step
# ---------------------------------------------------------------------------

def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the MSE loss (grad attached).

    The FL runtime is responsible for loss.backward() and optimizer.step().
    This function only calls optimizer.zero_grad() to clear stale gradients
    before the forward pass, matching the original training loop.

    Label shape is broadcast-safe against Regressor's [B, 1] output:
    a rank-1 label tensor [B] is unsqueezed to [B, 1] automatically.
    """
    device = next(model.parameters()).device

    inputs = batch[0].to(device)
    labels = batch[1].to(device).float()

    optimizer.zero_grad()

    outputs = model(inputs)  # shape: [B, out_shape] e.g. [B, 1]

    # Align label rank to model output rank ([B] → [B, 1] for out_shape=1).
    while labels.ndim < outputs.ndim:
        labels = labels.unsqueeze(-1)

    loss_function = torch.nn.MSELoss()
    loss = loss_function(outputs, labels)
    # Do NOT call loss.backward() or optimizer.step() — handled by FL runtime.
    return loss
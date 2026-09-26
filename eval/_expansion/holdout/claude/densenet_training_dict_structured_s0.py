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

import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd

# Additional imports required by the FL interface
from torch.utils.data import Dataset as TorchDataset, random_split

# ---------------------------------------------------------------------------
# Image catalogue preserved verbatim from the original script
# ---------------------------------------------------------------------------

_IMAGE_NAMES = [
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
# Internal dataset helpers
# ---------------------------------------------------------------------------

class _MonaiIXIDataset(TorchDataset):
    """Wraps a MONAI Dataset (with MONAI transforms) as a plain PyTorch Dataset
    so that random_split can index into it without touching the transform pipeline."""

    def __init__(self, data_dicts: list, transform: Compose) -> None:
        self._ds = monai.data.Dataset(data=data_dicts, transform=transform)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


class _SyntheticIXIDataset(TorchDataset):
    """Synthetic stand-in: random 3-D volumes (C=1, 96³) with binary labels.

    Only instantiated when config['allow_synthetic_data'] is True.
    """

    def __init__(self, length: int = 20) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        img = torch.randn(1, 96, 96, 96)
        label = torch.randint(0, 2, ()).long()
        return {"img": img, "label": label}


# ---------------------------------------------------------------------------
# FL public interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the MONAI DenseNet121 classifier.

    Reads constructor arguments from config.get("model_kwargs", {}).
    Defaults reproduce the original script exactly.
    """
    kwargs = config.get("model_kwargs", {})
    model = monai.networks.nets.DenseNet121(
        spatial_dims=kwargs.get("spatial_dims", 3),
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 2),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    Data discovery
    --------------
    All 20 IXI-T1 NIfTI files are expected under config["data_path"].
    If any file is missing:
      - config["allow_synthetic_data"] == True  → use _SyntheticIXIDataset.
      - config["allow_synthetic_data"] == False → raise FileNotFoundError.

    Train / val splitting
    ---------------------
    random_split (seed 42, 80 / 20) is applied to the full dataset so that
    both splits are drawn from the same underlying data without overlap.
    Split-specific MONAI transforms (RandRotate90d only for train) are applied
    through _MonaiIXIDataset before random_split indexes into it.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get(
        "data_path",
        os.sep.join([".", "workspace", "data", "medical", "ixi", "IXI-T1"]),
    )

    image_paths = [os.path.join(data_path, name) for name in _IMAGE_NAMES]
    data_available = all(os.path.isfile(p) for p in image_paths)

    if data_available:
        train_transforms = Compose(
            [
                LoadImaged(keys=["img"], ensure_channel_first=True),
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=(96, 96, 96)),
                RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["img"], ensure_channel_first=True),
                ScaleIntensityd(keys=["img"]),
                Resized(keys=["img"], spatial_size=(96, 96, 96)),
            ]
        )
        data_dicts = [
            {"img": img_path, "label": int(lbl)}
            for img_path, lbl in zip(image_paths, _LABELS)
        ]
        # Choose transforms appropriate for the requested split so that
        # augmentation is only applied to training samples.
        transform = train_transforms if split == "train" else val_transforms
        full_dataset: TorchDataset = _MonaiIXIDataset(data_dicts, transform)
    else:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"IXI-T1 NIfTI files not found under '{data_path}'. "
                "Ensure the dataset is present at that path, or set "
                "config['allow_synthetic_data'] = True to permit synthetic data."
            )
        full_dataset = _SyntheticIXIDataset(length=20)

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val

    # Deterministic split: same seed produces consistent train / val indices
    # regardless of whether this function is called for "train" or "val".
    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    subset = train_subset if split == "train" else val_subset
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run a single forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.
    """
    device = next(model.parameters()).device

    inputs = batch["img"].to(device)
    # Cast labels to long (int64) — CrossEntropyLoss requires class indices.
    labels = batch["label"].to(device).long()

    loss_fn = torch.nn.CrossEntropyLoss()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    # loss.backward() and optimizer.step() are intentionally omitted.
    return loss
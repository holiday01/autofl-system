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

"""
Auto-generated FL client module.
Original script: MONAI DenseNet121 3-D brain MRI gender classification (IXI dataset)

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
)


# ── Dataset ─────────────────────────────────────────────────────────────

class SyntheticVolumeDataset(Dataset):
    """Synthetic fallback for unit-testing without real NIfTI files."""

    def __init__(self, n: int = 20, spatial_size=(96, 96, 96), num_classes: int = 2):
        self.n = n
        self.spatial_size = spatial_size
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.randn(1, *self.spatial_size, dtype=torch.float32)
        label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        return {"img": img, "label": label}


# ── Transforms ───────────────────────────────────────────────────────────

def _build_transforms(split: str, spatial_size: tuple) -> Compose:
    base = [
        LoadImaged(keys=["img"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=spatial_size),
    ]
    if split == "train":
        base.append(RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]))
    return Compose(base)


# ── FL Interface ──────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return monai.networks.nets.DenseNet121(
        spatial_dims=kwargs.get("spatial_dims", 3),
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 2),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  2))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory",  True)

    spatial_size = tuple(config.get("spatial_size", (96, 96, 96)))
    num_classes  = config.get("model_kwargs", {}).get("out_channels", 2)

    file_list = config.get("train_files" if split == "train" else "val_files", None)

    if file_list is None:
        data_path = config.get("data_path", "")
        if os.path.isdir(data_path):
            all_images = sorted(
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith(".nii.gz")
            )
            all_labels = config.get("labels", [0] * len(all_images))
            n_train = int(len(all_images) * config.get("train_ratio", 0.5))
            imgs, labs = (
                (all_images[:n_train], all_labels[:n_train])
                if split == "train"
                else (all_images[n_train:], all_labels[n_train:])
            )
            file_list = [{"img": img, "label": int(lab)} for img, lab in zip(imgs, labs)]

    if not file_list:
        dataset = SyntheticVolumeDataset(
            n=config.get("synthetic_n", 20),
            spatial_size=spatial_size,
            num_classes=num_classes,
        )
    else:
        transforms = _build_transforms(split, spatial_size)
        dataset = monai.data.Dataset(data=file_list, transform=transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list | dict,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, dict):
        inputs  = batch["img"].to(device)   if isinstance(batch["img"],   torch.Tensor) else batch["img"]
        targets = batch["label"].to(device) if isinstance(batch["label"], torch.Tensor) else torch.tensor(batch["label"], device=device)
    elif isinstance(batch, (list, tuple)):
        inputs, targets = batch[0].to(device), batch[1].to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    return loss
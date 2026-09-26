"""
Auto-generated FL client module.
Original script: MONAI 3-D brain MRI gender classification (IXI-T1 dataset).

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT:
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
from torch.utils.data import random_split

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirst, RandRotate90, Resize, ScaleIntensity


# ── Dataset helpers ──────────────────────────────────────────────────────────

def _build_transforms(spatial_size: tuple, split: str):
    base = [ScaleIntensity(), EnsureChannelFirst(), Resize(spatial_size)]
    if split == "train":
        base.append(RandRotate90())
    return Compose(base)


def _collect_image_label_lists(data_path: str, image_files: list[str], labels: list[int]):
    """
    Returns (image_paths, label_array).
    If image_files is supplied explicitly, use them; otherwise scan data_path for .nii/.nii.gz.
    """
    if image_files:
        paths = [
            p if os.path.isabs(p) else os.path.join(data_path, p)
            for p in image_files
        ]
        return paths, np.array(labels, dtype=np.int64)

    paths, auto_labels = [], []
    if os.path.isdir(data_path):
        for label_idx, subdir in enumerate(sorted(os.listdir(data_path))):
            subpath = os.path.join(data_path, subdir)
            if os.path.isdir(subpath):
                for f in sorted(os.listdir(subpath)):
                    if f.endswith(".nii.gz") or f.endswith(".nii"):
                        paths.append(os.path.join(subpath, f))
                        auto_labels.append(label_idx)
    return paths, np.array(auto_labels, dtype=np.int64)


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    spatial_dims = kwargs.get("spatial_dims", 3)
    in_channels  = kwargs.get("in_channels", 1)
    out_channels = kwargs.get("out_channels", 2)
    return monai.networks.nets.DenseNet121(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory",  True)

    data_path   = config.get("data_path", ".")
    image_files = config.get("image_files", [])
    labels      = config.get("labels", [])
    spatial_size = tuple(config.get("spatial_size", [96, 96, 96]))
    val_ratio   = config.get("val_ratio", 0.5)
    seed        = config.get("seed", 42)

    all_images, all_labels = _collect_image_label_lists(data_path, image_files, labels)

    if len(all_images) == 0:
        raise ValueError(
            f"No NIfTI images found in '{data_path}' and no image_files provided in config."
        )

    n_val   = max(1, int(len(all_images) * val_ratio))
    n_train = len(all_images) - n_val

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_images))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    if split == "train":
        sel_idx = train_idx
    else:
        sel_idx = val_idx

    sel_images = [all_images[i] for i in sel_idx]
    sel_labels = all_labels[sel_idx]

    transforms = _build_transforms(spatial_size, split)
    dataset = ImageDataset(
        image_files=sel_images,
        labels=sel_labels,
        transform=transforms,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if targets.dtype == torch.float32 or targets.dtype == torch.float64:
        targets = targets.long()

    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    return loss
"""
Auto-generated FL client module.
Original script: MONAI 3-D classification example (IXI-T1 gender classification).

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
from torch.utils.data import DataLoader, TensorDataset, random_split

import monai
from monai.data import ImageDataset
from monai.transforms import Compose, EnsureChannelFirst, RandRotate90, Resize, ScaleIntensity


# ── Dataset helpers ─────────────────────────────────────────────────────────

def _discover_image_files(data_path: str):
    """Walk data_path and collect (.nii.gz / .nii) file paths + integer labels."""
    image_files, labels = [], []
    if not os.path.isdir(data_path):
        return image_files, labels
    for label_idx, subdir in enumerate(sorted(os.listdir(data_path))):
        subpath = os.path.join(data_path, subdir)
        if os.path.isdir(subpath):
            for fname in sorted(os.listdir(subpath)):
                if fname.endswith(".nii.gz") or fname.endswith(".nii"):
                    image_files.append(os.path.join(subpath, fname))
                    labels.append(label_idx)
    return image_files, labels


class _SyntheticVolumeDataset(TensorDataset):
    """In-memory synthetic 3-D volume dataset for smoke-testing."""

    def __init__(self, n: int = 40, spatial_size=(96, 96, 96), num_classes: int = 2, seed: int = 42):
        gen = torch.Generator().manual_seed(seed)
        X = torch.randn(n, 1, *spatial_size, generator=gen)
        y = torch.randint(0, num_classes, (n,), generator=gen)
        super().__init__(X, y)


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    spatial_dims  = kwargs.get("spatial_dims", 3)
    in_channels   = kwargs.get("in_channels", 1)
    out_channels  = kwargs.get("out_channels", 2)
    return monai.networks.nets.DenseNet121(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local         = config.get("local", {})
    batch_size    = local.get("batch_size",  config.get("batch_size",  2))
    num_workers   = local.get("num_workers", config.get("num_workers", 2))
    pin_memory    = local.get("pin_memory",  True)
    seed          = config.get("seed", 42)
    val_ratio     = config.get("val_ratio", 0.2)
    spatial_size  = tuple(config.get("spatial_size", [96, 96, 96]))
    num_classes   = config.get("num_classes", 2)
    data_path     = config.get("data_path", ".")

    image_files, labels = _discover_image_files(data_path)

    if not image_files:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No NIfTI files found under '{data_path}'. "
                "Set config['allow_synthetic_data'] = True to use synthetic data."
            )
        n_synth = config.get("synthetic_n", 40)
        full_dataset = _SyntheticVolumeDataset(
            n=n_synth, spatial_size=spatial_size, num_classes=num_classes, seed=seed
        )
    else:
        train_transforms = Compose([
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize(spatial_size),
            RandRotate90(),
        ])
        val_transforms = Compose([
            ScaleIntensity(),
            EnsureChannelFirst(),
            Resize(spatial_size),
        ])
        n_val   = max(1, int(len(image_files) * val_ratio))
        n_train = len(image_files) - n_val
        # deterministic split by index (no random_split on ImageDataset)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(image_files))
        train_idx, val_idx = idx[:n_train].tolist(), idx[n_train:].tolist()

        transform = train_transforms if split == "train" else val_transforms
        selected  = train_idx if split == "train" else val_idx
        sel_files  = [image_files[i] for i in selected]
        sel_labels = np.array([labels[i] for i in selected], dtype=np.int64)
        full_dataset = ImageDataset(
            image_files=sel_files,
            labels=sel_labels,
            transform=transform,
        )
        return DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    # synthetic path — use random_split
    n_val_syn   = max(1, int(len(full_dataset) * val_ratio))
    n_train_syn = len(full_dataset) - n_val_syn
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train_syn, n_val_syn],
        generator=torch.Generator().manual_seed(seed),
    )
    ds = train_ds if split == "train" else val_ds
    return DataLoader(
        ds,
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

    outputs = model(inputs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss
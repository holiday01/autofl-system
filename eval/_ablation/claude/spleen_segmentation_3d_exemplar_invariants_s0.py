"""
Auto-generated FL client module.
Original script: MONAI 3D UNet segmentation training script.

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
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import monai
from monai.data import list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)


class SyntheticSegDataset(Dataset):
    """Returns dict batches of (1, D, H, W) image/seg tensors for testing."""

    def __init__(self, n: int = 40, spatial_size: tuple = (96, 96, 96)):
        self.n = n
        self.spatial_size = spatial_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        D, H, W = self.spatial_size
        img = torch.randn(1, D, H, W, dtype=torch.float32)
        seg = torch.randint(0, 2, (1, D, H, W), dtype=torch.float32)
        return {"img": img, "seg": seg}


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return monai.networks.nets.UNet(
        spatial_dims=kwargs.get("spatial_dims", 3),
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 1),
        channels=kwargs.get("channels", (16, 32, 64, 128, 256)),
        strides=kwargs.get("strides", (2, 2, 2, 2)),
        num_res_units=kwargs.get("num_res_units", 2),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory", True)

    data_path = config.get("data_path", ".")
    allow_synthetic = config.get("allow_synthetic_data", False)
    spatial_size = config.get("spatial_size", [96, 96, 96])

    images = sorted(glob(os.path.join(data_path, "img*.nii.gz")))
    segs   = sorted(glob(os.path.join(data_path, "seg*.nii.gz")))

    if not images or not segs:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"No NIfTI image/segmentation pairs found in '{data_path}'. "
                "Set config['allow_synthetic_data'] = True to use synthetic data."
            )
        n_synthetic = config.get("n_synthetic", 40)
        full_dataset = SyntheticSegDataset(n=n_synthetic, spatial_size=tuple(spatial_size))
        val_ratio = config.get("val_ratio", 0.2)
        n_val = max(1, int(len(full_dataset) * val_ratio))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.get("seed", 42)),
        )
        ds = train_ds if split == "train" else val_ds
        return DataLoader(
            ds,
            batch_size=batch_size if split == "train" else 1,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    all_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
    val_ratio = config.get("val_ratio", 0.2)
    n_val = max(1, int(len(all_files) * val_ratio))
    train_files = all_files[:-n_val]
    val_files   = all_files[-n_val:]

    train_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys="img"),
        RandCropByPosNegLabeld(
            keys=["img", "seg"], label_key="seg",
            spatial_size=spatial_size, pos=1, neg=1, num_samples=4,
        ),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        ScaleIntensityd(keys="img"),
    ])

    files = train_files if split == "train" else val_files
    transforms = train_transforms if split == "train" else val_transforms
    ds = monai.data.Dataset(data=files, transform=transforms)

    return DataLoader(
        ds,
        batch_size=batch_size if split == "train" else 1,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list | dict,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, dict):
        inputs  = batch["img"].to(device) if isinstance(batch["img"], torch.Tensor) else batch["img"]
        targets = batch["seg"].to(device) if isinstance(batch["seg"], torch.Tensor) else batch["seg"]
    elif isinstance(batch, (list, tuple)):
        inputs  = batch[0].to(device) if isinstance(batch[0], torch.Tensor) else batch[0]
        targets = batch[1].to(device) if isinstance(batch[1], torch.Tensor) else batch[1]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    loss_fn = monai.losses.DiceLoss(sigmoid=True)
    loss = loss_fn(outputs, targets)
    return loss
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity


def build_model(config: dict) -> torch.nn.Module:
    kwargs = config.get("model_kwargs", {})
    return monai.networks.nets.DenseNet121(
        spatial_dims=kwargs.get("spatial_dims", 3),
        in_channels=kwargs.get("in_channels", 1),
        out_channels=kwargs.get("out_channels", 2),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    val_fraction = config.get("val_fraction", 0.2)
    seed = config.get("split_seed", 42)

    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    image_files = None
    labels = None

    if os.path.isdir(data_path):
        candidates = sorted(
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".nii.gz") or f.endswith(".nii")
        )
        labels_npy = os.path.join(data_path, "labels.npy")
        labels_csv = os.path.join(data_path, "labels.csv")

        if candidates and os.path.isfile(labels_npy):
            image_files = candidates
            labels = np.load(labels_npy)
        elif candidates and os.path.isfile(labels_csv):
            import csv
            with open(labels_csv) as fh:
                labels = np.array([int(row[0]) for row in csv.reader(fh)], dtype=np.int64)
            image_files = candidates

    if image_files is None or labels is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No NIfTI images with a labels.npy or labels.csv found in '{data_path}'. "
                "Set config['allow_synthetic_data']=True to use synthetic data for testing."
            )
        n_samples = 20
        imgs = torch.randn(n_samples, 1, 96, 96, 96)
        lbls = torch.randint(0, 2, (n_samples,))
        full_ds = TensorDataset(imgs, lbls)
        n_val = max(1, int(n_samples * val_fraction))
        n_train = n_samples - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )
        chosen = train_ds if split == "train" else val_ds
        return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))

    transforms = train_transforms if split == "train" else val_transforms
    full_ds = ImageDataset(image_files=image_files, labels=labels, transform=transforms)

    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    chosen = train_ds if split == "train" else val_ds
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    inputs = batch[0].to(device)
    labels = batch[1].to(device)
    outputs = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    return loss
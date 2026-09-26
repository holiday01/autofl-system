"""
Auto-generated FL client module.
Original script: monai_densenet_classification3d.py

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor
"""
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity

# ── Original source (unchanged) ────────────────────────────────────────
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

from torch.utils.tensorboard import SummaryWriter


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
    # the path of ixi IXI-T1 dataset
    data_path = os.sep.join([".", "workspace", "data", "medical", "ixi", "IXI-T1"])
    images = [
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
    images = [os.sep.join([data_path, f]) for f in images]

    # 2 binary labels for gender classification: man and woman
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

    # Define transforms
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    # Define image dataset, data loader
    check_ds = ImageDataset(image_files=images, labels=labels, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())
    im, label = monai.utils.misc.first(check_loader)
    print(type(im), im.shape, label)

    # create a training data loader
    train_ds = ImageDataset(image_files=images[:10], labels=labels[:10], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = ImageDataset(image_files=images[-10:], labels=labels[-10:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
                metric = num_correct / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate DenseNet121. Override __init__ kwargs via config['model_kwargs']."""
    kwargs = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
    }
    kwargs.update(config.get("model_kwargs", {}))
    return monai.networks.nets.DenseNet121(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for the requested split.

    config must supply either:
      - 'image_files' (list[str]) and 'labels' (list[int] or np.ndarray), or
      - 'data_path' (str) for flat-directory discovery (filenames ending in .nii / .nii.gz,
        labels provided via 'labels' or defaulting to 0).

    Optional config keys:
      'val_ratio'       float  fraction held out for validation (default 0.1)
      'resize_spatial'  tuple  spatial size for Resize (default (96, 96, 96))
      'seed'            int    random seed for the train/val split (default 42)
      'local'           dict   per-client overrides: batch_size, num_workers, pin_memory
    """
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    resize_spatial = tuple(config.get("resize_spatial", (96, 96, 96)))

    train_transforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize(resize_spatial),
        RandRotate90(),
    ])
    val_transforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize(resize_spatial),
    ])

    # Resolve image files and labels
    if "image_files" in config:
        image_files = config["image_files"]
        labels = np.array(config["labels"], dtype=np.int64)
    else:
        data_path = config.get("data_path", ".")
        image_files = sorted(
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        )
        labels = np.array(config.get("labels", [0] * len(image_files)), dtype=np.int64)

    # Split into train / val
    n_total = len(image_files)
    val_ratio = config.get("val_ratio", 0.1)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    rng = np.random.default_rng(config.get("seed", 42))
    indices = rng.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    if split == "train":
        ds = ImageDataset(
            image_files=[image_files[i] for i in train_idx],
            labels=labels[train_idx],
            transform=train_transforms,
        )
    else:
        ds = ImageDataset(
            image_files=[image_files[i] for i in val_idx],
            labels=labels[val_idx],
            transform=val_transforms,
        )

    return DataLoader(
        ds,
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
    Perform one forward (+ optionally backward) step.
    If optimizer is None (preflight forward-only check), skip backward.
    """
    local   = config.get("local", {})
    use_amp = local.get("use_amp", False)
    device  = next(model.parameters()).device

    # Move batch to device — supports (inputs, targets) or dict batches
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

    with torch.autocast(
        device_type=device.type if hasattr(device, "type") else str(device),
        enabled=use_amp,
    ):
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)


    return loss
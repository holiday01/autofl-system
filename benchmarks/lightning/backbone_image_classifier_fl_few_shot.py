"""
Auto-generated FL client module.
Original script: backbone_image_classifier.py

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

# ── Original source (unchanged) ────────────────────────────────────────
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MNIST backbone image classifier example."""

from os import path
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

try:
    from torchvision import transforms
    from torchvision.datasets import MNIST
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate Backbone. Override __init__ kwargs via config['model_kwargs']."""
    kwargs = config.get("model_kwargs", {})
    return Backbone(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for the requested split ('train', 'val', or 'test').
    Expects config to have 'data_path' (MNIST root) and optionally 'val_size'.
    Client-local batch_size and num_workers are read from config['local'].
    """
    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError("torchvision is required for MNIST data loading.")

    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    data_path = config.get("data_path", DATASETS_PATH)
    transform = transforms.ToTensor()

    if split == "test":
        dataset = MNIST(data_path, train=False, download=True, transform=transform)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    full_train = MNIST(data_path, train=True, download=True, transform=transform)
    val_size   = config.get("val_size", 5000)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
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

    with torch.autocast(device_type=device.type if hasattr(device, "type") else str(device),
                        enabled=use_amp):
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

    if optimizer is not None:
        if use_amp:
            scaler = getattr(train_step, "_scaler", None)
            if scaler is None:
                train_step._scaler = torch.cuda.amp.GradScaler()
                scaler = train_step._scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    return loss.detach()
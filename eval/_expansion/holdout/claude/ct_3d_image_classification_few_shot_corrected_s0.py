"""
Auto-generated FL client module.
Original script: 3D image classification from CT scans (Keras → PyTorch)

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
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split


# ── Preprocessing ─────────────────────────────────────────────────────

def _read_nifti_file(filepath: str) -> np.ndarray:
    import nibabel as nib
    scan = nib.load(filepath)
    return scan.get_fdata()


def _normalize(volume: np.ndarray) -> np.ndarray:
    volume = np.clip(volume, -1000, 400)
    volume = (volume - (-1000)) / (400 - (-1000))
    return volume.astype(np.float32)


def _resize_volume(img: np.ndarray,
                   desired_width: int = 128,
                   desired_height: int = 128,
                   desired_depth: int = 64) -> np.ndarray:
    from scipy import ndimage
    w_factor = desired_width  / img.shape[0]
    h_factor = desired_height / img.shape[1]
    d_factor = desired_depth  / img.shape[2]
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (w_factor, h_factor, d_factor), order=1)
    return img


def _process_scan(path: str) -> np.ndarray:
    volume = _read_nifti_file(path)
    volume = _normalize(volume)
    volume = _resize_volume(volume)
    return volume


def _rotate_augment(volume: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    angle = random.choice([-20, -10, -5, 5, 10, 20])
    volume = ndimage.rotate(volume, angle, reshape=False)
    return np.clip(volume, 0.0, 1.0)


# ── Dataset ───────────────────────────────────────────────────────────

class CTScanDataset(Dataset):
    """
    Binary CT scan dataset (0 = normal CT-0, 1 = abnormal CT-23).
    Expects data_root/{CT-0,CT-23}/*.nii[.gz]; falls back to synthetic data.
    """

    def __init__(self, root: str, augment: bool = False):
        self.augment = augment
        self._preloaded = None

        normal_dir   = os.path.join(root, "CT-0")
        abnormal_dir = os.path.join(root, "CT-23")

        if os.path.isdir(normal_dir) and os.path.isdir(abnormal_dir):
            self.samples = []
            for fname in sorted(os.listdir(normal_dir)):
                if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                    self.samples.append((os.path.join(normal_dir, fname), 0))
            for fname in sorted(os.listdir(abnormal_dir)):
                if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                    self.samples.append((os.path.join(abnormal_dir, fname), 1))
        else:
            # synthetic fallback for unit-testing without real CT data
            n = 200
            self._preloaded = [
                (np.random.randn(128, 128, 64).astype(np.float32), i % 2)
                for i in range(n)
            ]
            self.samples = list(range(n))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self._preloaded is not None:
            volume, label = self._preloaded[idx]
        else:
            path, label = self.samples[idx]
            volume = _process_scan(path)

        if self.augment:
            volume = _rotate_augment(volume)

        # (W, H, D) → (1, W, H, D) channel dimension for Conv3d
        x = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


# ── Model ─────────────────────────────────────────────────────────────

class _Conv3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class CNN3D(nn.Module):
    """3D CNN for binary CT scan classification (pneumonia detection)."""

    def __init__(self, in_channels: int = 1, dense_units: int = 512, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _Conv3DBlock(in_channels, 64),
            _Conv3DBlock(64, 64),
            _Conv3DBlock(64, 128),
            _Conv3DBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)


# ── FL Interface ──────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return CNN3D(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  2))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory",  True)

    data_path = config.get("data_path", ".")
    val_ratio = config.get("val_ratio", 0.3)
    seed      = config.get("seed", 42)

    # Determine split indices on a no-augmentation reference dataset.
    ref = CTScanDataset(root=data_path, augment=False)
    n_val   = max(1, int(len(ref) * val_ratio))
    n_train = len(ref) - n_val
    train_sub, val_sub = random_split(
        ref, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    if split == "train":
        use_augment = config.get("augment", True)
        aug_ds = CTScanDataset(root=data_path, augment=use_augment)
        ds = Subset(aug_ds, train_sub.indices)
    else:
        ds = val_sub

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
    loss = nn.BCELoss()(outputs, targets)
    return loss
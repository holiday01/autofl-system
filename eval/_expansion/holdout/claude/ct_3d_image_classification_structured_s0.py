import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

try:
    import nibabel as nib
    from scipy import ndimage
    _HAS_NIBABEL = True
except ImportError:
    _HAS_NIBABEL = False

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


# ── Preprocessing helpers (preserved from original) ───────────────────────────

def read_nifti_file(filepath):
    """Read and load volume"""
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min_hu = -1000
    max_hu = 400
    volume[volume < min_hu] = min_hu
    volume[volume > max_hu] = max_hu
    volume = (volume - min_hu) / (max_hu - min_hu)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    volume = read_nifti_file(path)
    volume = normalize(volume)
    volume = resize_volume(volume)
    return volume


def _rotate_numpy(volume):
    """Rotate the volume by a few degrees (NumPy-based, no TF dependency)."""
    angles = [-20, -10, -5, 5, 10, 20]
    angle = random.choice(angles)
    volume = ndimage.rotate(volume, angle, reshape=False)
    volume = np.clip(volume, 0.0, 1.0).astype("float32")
    return volume


# ── Datasets ──────────────────────────────────────────────────────────────────

class CTScanDataset(Dataset):
    """Loads NIfTI CT scans and returns (1, D, H, W) float32 tensors."""

    def __init__(self, scan_paths, labels):
        self.scan_paths = scan_paths
        self.labels = labels

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        # process_scan → (W=128, H=128, D=64)
        volume = process_scan(self.scan_paths[idx])
        # Reorder to PyTorch NCDHW: (1, D, H, W) = (1, 64, 128, 128)
        volume = np.transpose(volume, (2, 0, 1))[np.newaxis]  # (1, D, W, H)
        label = np.float32(self.labels[idx])
        return torch.from_numpy(volume.copy()), torch.tensor(label)


class _AugmentSubset(Dataset):
    """Wraps a Subset and applies random-rotation augmentation on the fly."""

    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        volume, label = self.subset[idx]
        # volume: (1, D, H, W)  →  squeeze to (D, H, W), rotate, unsqueeze back
        vol_np = volume.squeeze(0).numpy()
        vol_np = _rotate_numpy(vol_np)
        return torch.from_numpy(vol_np).unsqueeze(0), label


# ── Model: Keras get_model() → PyTorch CNN3D ─────────────────────────────────

class CNN3D(nn.Module):
    """
    3D convolutional neural network – exact PyTorch equivalent of the
    Keras model returned by get_model(width, height, depth).

    Input shape : (B, 1, D, H, W)  e.g. (B, 1, 64, 128, 128)
    Output shape: (B, 1)  – sigmoid probability
    """

    def __init__(self, width=128, height=128, depth=64):
        super().__init__()
        # Store for reference; Conv3d is spatially agnostic at build time.
        self.width = width
        self.height = height
        self.depth = depth

        self.encoder = nn.Sequential(
            # ── Block 1 ─────────────────────────────────────────────────────
            nn.Conv3d(1, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(64),
            # ── Block 2 ─────────────────────────────────────────────────────
            nn.Conv3d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(64),
            # ── Block 3 ─────────────────────────────────────────────────────
            nn.Conv3d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(128),
            # ── Block 4 ─────────────────────────────────────────────────────
            nn.Conv3d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(256),
        )
        # GlobalAveragePooling3D equivalent
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)


# ── FL API ────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the 3-D CNN.

    Recognised model_kwargs:
        width  (int, default 128)
        height (int, default 128)
        depth  (int, default  64)
    """
    kwargs = config.get("model_kwargs", {})
    return CNN3D(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Expected on-disk layout:
        <data_path>/MosMedData/CT-0/   – normal  CT scans  (*.nii / *.nii.gz)
        <data_path>/MosMedData/CT-23/  – abnormal CT scans (*.nii / *.nii.gz)

    Config keys
    -----------
    data_path              : str   root directory  (default ".")
    local.batch_size       : int   mini-batch size (default 2)
    allow_synthetic_data   : bool  fall back to torch.randn data when real
                                   data or nibabel are unavailable (default False)
    """
    batch_size = config.get("local", {}).get("batch_size", 2)
    data_path = config.get("data_path", ".")

    normal_dir = os.path.join(data_path, "MosMedData", "CT-0")
    abnormal_dir = os.path.join(data_path, "MosMedData", "CT-23")

    real_data_ok = (
        _HAS_NIBABEL
        and os.path.isdir(normal_dir)
        and os.path.isdir(abnormal_dir)
    )

    if not real_data_ok:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"CT scan directories not found under '{data_path}/MosMedData/'. "
                "Expected sub-directories 'CT-0' (normal) and 'CT-23' (abnormal) "
                "containing *.nii or *.nii.gz files. "
                "Install nibabel (`pip install nibabel`) and ensure the data is present, "
                "or set config['allow_synthetic_data']=True to use synthetic tensors "
                "for smoke-testing only."
            )
        # ── Synthetic fallback (testing / CI only) ────────────────────────
        n_synth = 20
        # Shape mirrors real pre-processed volumes: (1, D=64, H=128, W=128)
        X_synth = torch.randn(n_synth, 1, 64, 128, 128)
        y_synth = torch.randint(0, 2, (n_synth,)).float()
        full_ds = TensorDataset(X_synth, y_synth)
        n_train = int(0.7 * n_synth)
        n_val = n_synth - n_train
        train_sub, val_sub = random_split(full_ds, [n_train, n_val])
        chosen = train_sub if split == "train" else val_sub
        return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))

    # ── Real data ─────────────────────────────────────────────────────────────
    def _gather(directory):
        return sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        )

    normal_paths = _gather(normal_dir)
    abnormal_paths = _gather(abnormal_dir)

    all_paths = normal_paths + abnormal_paths
    all_labels = [0] * len(normal_paths) + [1] * len(abnormal_paths)

    if len(all_paths) == 0:
        raise FileNotFoundError(
            f"No *.nii / *.nii.gz files found in '{normal_dir}' or '{abnormal_dir}'."
        )

    full_ds = CTScanDataset(all_paths, all_labels)

    n_train = max(1, int(0.7 * len(full_ds)))
    n_val = len(full_ds) - n_train
    train_sub, val_sub = random_split(full_ds, [n_train, n_val])

    if split == "train":
        # Wrap training split with random-rotation augmentation
        chosen = _AugmentSubset(train_sub)
    else:
        chosen = val_sub

    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(
    model: nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Single forward pass.  Returns the loss tensor with grad attached.
    Does NOT call loss.backward() or optimizer.step() – the FL runtime
    is responsible for both.
    """
    device = next(model.parameters()).device

    volumes, labels = batch
    volumes = volumes.to(device)                  # (B, 1, D, H, W)
    labels = labels.to(device).view(-1, 1).float()  # (B, 1)

    preds = model(volumes)                        # (B, 1), sigmoid output
    loss = F.binary_cross_entropy(preds, labels)
    return loss
import os
import glob
import warnings
warnings.filterwarnings("ignore")

import shutil
import numpy as np

try:
    import kagglehub
except ImportError:
    kagglehub = None

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from monai.networks.nets import SwinUNETR as _MonaiSwinUNETR
    from monai.transforms import (
        Compose,
        CropForegroundd,
        NormalizeIntensityd,
        RandFlipd,
        RandShiftIntensityd,
        RandSpatialCropd,
    )
    from monai.losses import DiceCELoss as _MonaiDiceCELoss
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


# ─────────────────────────────────────────────────────────────────────────────
# Preserved architecture
# Original: medicai.models.SwinUNETR(encoder_name="swin_tiny_v2",
#               input_shape=(96, 96, 96, 4), num_classes=3, classifier_activation=None)
# Converted from Keras/medicai → equivalent PyTorch/MONAI SwinUNETR.
# swin_tiny_v2  ≈  feature_size=24 in MONAI SwinUNETR.
# ─────────────────────────────────────────────────────────────────────────────

class SwinUNETR(nn.Module):
    """
    PyTorch/MONAI equivalent of medicai SwinUNETR(encoder_name='swin_tiny_v2').

    Input  : (B, in_channels,  D, H, W) — channel-first
    Output : (B, out_channels, D, H, W) — raw logits (no activation)
    """

    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=4,
        out_channels=3,
        feature_size=24,
        use_checkpoint=True,
        **kwargs,
    ):
        super().__init__()
        if not HAS_MONAI:
            raise ImportError(
                "monai is required.  Install with:  pip install 'monai[all]'"
            )
        self._net = _MonaiSwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


# ─────────────────────────────────────────────────────────────────────────────
# BraTS multi-channel label conversion (preserved from original script)
# ─────────────────────────────────────────────────────────────────────────────

class ConvertToMultiChannelBasedOnBratsClasses:
    """
    Convert labels to multi channels based on BRATS classes.

    Label definitions:
    - 1: necrotic and non-enhancing tumor core
    - 2: peritumoral edema
    - 4: GD-enhancing tumor

    Output channels:
    - Channel 0 (TC): Tumor core  — label ∈ {1, 4}
    - Channel 1 (WT): Whole tumor — label ∈ {1, 2, 4}
    - Channel 2 (ET): Enhancing   — label == 4

    Input:  numpy array or torch tensor, shape (D, H, W)
    Output: torch.float32 tensor,        shape (3, D, H, W)
    """

    def __call__(self, label):
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(np.ascontiguousarray(label))
        label = label.float()
        tc = (label == 1) | (label == 4)
        wt = tc | (label == 2)
        et = label == 4
        return torch.stack([tc.float(), wt.float(), et.float()], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# TFRecord parsing — preserves parse_tfrecord_fn + rearrange_shape from original
# ─────────────────────────────────────────────────────────────────────────────

def _parse_and_rearrange(raw_example):
    """
    Parse one serialised TFRecord example.
    Returns (image_np, label_np) with
        image_np : float32 ndarray (D, H, W, 4)   channel-last
        label_np : float32 ndarray (D, H, W)       integer values {0,1,2,4}
    """
    feature_description = {
        "flair_raw":      tf.io.FixedLenFeature([], tf.string),
        "t1_raw":         tf.io.FixedLenFeature([], tf.string),
        "t1ce_raw":       tf.io.FixedLenFeature([], tf.string),
        "t2_raw":         tf.io.FixedLenFeature([], tf.string),
        "label_raw":      tf.io.FixedLenFeature([], tf.string),
        "flair_shape":    tf.io.FixedLenFeature([3], tf.int64),
        "t1_shape":       tf.io.FixedLenFeature([3], tf.int64),
        "t1ce_shape":     tf.io.FixedLenFeature([3], tf.int64),
        "t2_shape":       tf.io.FixedLenFeature([3], tf.int64),
        "label_shape":    tf.io.FixedLenFeature([3], tf.int64),
        "flair_affine":   tf.io.FixedLenFeature([16], tf.float32),
        "t1_affine":      tf.io.FixedLenFeature([16], tf.float32),
        "t1ce_affine":    tf.io.FixedLenFeature([16], tf.float32),
        "t2_affine":      tf.io.FixedLenFeature([16], tf.float32),
        "label_affine":   tf.io.FixedLenFeature([16], tf.float32),
        "flair_pixdim":   tf.io.FixedLenFeature([8], tf.float32),
        "t1_pixdim":      tf.io.FixedLenFeature([8], tf.float32),
        "t1ce_pixdim":    tf.io.FixedLenFeature([8], tf.float32),
        "t2_pixdim":      tf.io.FixedLenFeature([8], tf.float32),
        "label_pixdim":   tf.io.FixedLenFeature([8], tf.float32),
        "flair_filename": tf.io.FixedLenFeature([], tf.string),
        "t1_filename":    tf.io.FixedLenFeature([], tf.string),
        "t1ce_filename":  tf.io.FixedLenFeature([], tf.string),
        "t2_filename":    tf.io.FixedLenFeature([], tf.string),
        "label_filename": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(raw_example, feature_description)

    flair = tf.reshape(tf.io.decode_raw(ex["flair_raw"], tf.float32), ex["flair_shape"])
    t1    = tf.reshape(tf.io.decode_raw(ex["t1_raw"],    tf.float32), ex["t1_shape"])
    t1ce  = tf.reshape(tf.io.decode_raw(ex["t1ce_raw"],  tf.float32), ex["t1ce_shape"])
    t2    = tf.reshape(tf.io.decode_raw(ex["t2_raw"],    tf.float32), ex["t2_shape"])
    label = tf.reshape(tf.io.decode_raw(ex["label_raw"], tf.float32), ex["label_shape"])

    # Stack modalities → (W, H, D, C)
    image = tf.concat(
        [flair[..., None], t1[..., None], t1ce[..., None], t2[..., None]], axis=-1
    )
    # rearrange_shape: (W, H, D, C) → (D, H, W, C) and (W, H, D) → (D, H, W)
    image = tf.transpose(image, perm=[2, 1, 0, 3])
    label = tf.transpose(label, perm=[2, 1, 0])

    return image.numpy(), label.numpy()


def _load_all_samples(tfrec_files):
    """Eagerly load every record from a list of TFRecord shards."""
    samples = []
    for path in tfrec_files:
        ds = tf.data.TFRecordDataset(path)
        for raw in ds:
            samples.append(_parse_and_rearrange(raw))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset wrappers
# ─────────────────────────────────────────────────────────────────────────────

class BraTSRawDataset(Dataset):
    """
    Thin wrapper around a list of (image_np, label_np) tuples loaded from
    TFRecord shards.  No transforms applied; used as the source for random_split.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]   # (image_np, label_np)


class BraTSTransformDataset(Dataset):
    """
    Wraps a Subset of BraTSRawDataset and applies split-appropriate MONAI
    transforms (mirroring the original train_transformation / val_transformation).

    image_np (D, H, W, C) → permute → (C, D, H, W) channel-first torch tensor
    label_np (D, H, W)    → ConvertToMultiChannelBasedOnBratsClasses
                          → (3, D, H, W) binary torch tensor
    """

    def __init__(self, subset, split="train", roi_size=(96, 96, 96)):
        self.subset = subset
        self.split  = split
        self.label_converter = ConvertToMultiChannelBasedOnBratsClasses()

        if HAS_MONAI:
            if split == "train":
                self.transforms = Compose([
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                    ),
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=roi_size,
                        random_size=False,
                    ),
                    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
                    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=1.0),
                ])
            else:
                self.transforms = Compose([
                    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                ])
        else:
            self.transforms = None

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image_np, label_np = self.subset[idx]

        # (D, H, W, C) → (C, D, H, W)
        image = torch.from_numpy(np.ascontiguousarray(image_np)).permute(3, 0, 1, 2).float()
        # (D, H, W)    → (3, D, H, W)
        label = self.label_converter(label_np)

        if self.transforms is not None:
            data_dict = {"image": image, "label": label}
            data_dict = self.transforms(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]

        return image, label


class _SyntheticBraTSDataset(Dataset):
    """Synthetic data fallback — only instantiated when allow_synthetic_data=True."""

    def __init__(self, size=32, in_channels=4, out_channels=3, roi_size=(96, 96, 96)):
        self.size        = size
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.roi_size     = roi_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(self.in_channels, *self.roi_size)
        label = torch.randint(
            0, 2, (self.out_channels, *self.roi_size), dtype=torch.float32
        )
        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# FL Client API
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the SwinUNETR model.

    Converted from medicai (Keras) SwinUNETR(encoder_name='swin_tiny_v2',
    input_shape=(96,96,96,4), num_classes=3) to MONAI (PyTorch) SwinUNETR.

    config["model_kwargs"] keys:
        img_size       : tuple[int, int, int]   default (96, 96, 96)
        in_channels    : int                    default 4
        out_channels   : int                    default 3
        feature_size   : int                    default 24  (swin_tiny_v2)
        use_checkpoint : bool                   default True
    """
    kw             = config.get("model_kwargs", {})
    img_size       = tuple(kw.get("img_size",       (96, 96, 96)))
    in_channels    = int(kw.get("in_channels",      4))
    out_channels   = int(kw.get("out_channels",     3))
    feature_size   = int(kw.get("feature_size",     24))
    use_checkpoint = bool(kw.get("use_checkpoint",  True))

    return SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Reads BraTS TFRecord shards (training_shard_*.tfrec) from config["data_path"].
    Uses random_split to produce train / val subsets from the full sample list.

    Raises FileNotFoundError when no real data is found and
    config["allow_synthetic_data"] is False.

    config keys:
        data_path                : str   directory containing *.tfrec files
        local.batch_size         : int   default 1  (3-D volumes are memory-heavy)
        local.num_workers        : int   default 0
        val_fraction             : float default 0.2
        roi_size                 : list  default [96, 96, 96]
        allow_synthetic_data     : bool  default False
        model_kwargs.in_channels : int   default 4
        model_kwargs.out_channels: int   default 3
    """
    local_cfg    = config.get("local", {})
    batch_size   = int(local_cfg.get("batch_size",  1))
    num_workers  = int(local_cfg.get("num_workers", 0))
    data_path    = config.get("data_path", ".")
    roi_size     = tuple(int(x) for x in config.get("roi_size", [96, 96, 96]))
    val_fraction = float(config.get("val_fraction", 0.2))

    kw           = config.get("model_kwargs", {})
    in_channels  = int(kw.get("in_channels",  4))
    out_channels = int(kw.get("out_channels", 3))

    # ── Attempt to load real TFRecord data ───────────────────────────────────
    tfrec_files = sorted(
        glob.glob(os.path.join(data_path, "training_shard_*.tfrec"))
    )
    samples = None

    if tfrec_files and HAS_TF:
        try:
            samples = _load_all_samples(tfrec_files)
        except Exception as exc:
            warnings.warn(f"[build_dataloader] TFRecord loading failed: {exc}")
            samples = None

    # ── Synthetic fallback (must be explicitly enabled) ───────────────────────
    if samples is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No BraTS TFRecord files (training_shard_*.tfrec) found under "
                f"'{data_path}', or TensorFlow is unavailable to read them.  "
                "Provide real data at config['data_path'] or set "
                "config['allow_synthetic_data'] = True to use synthetic tensors."
            )
        full_ds = _SyntheticBraTSDataset(
            size=32,
            in_channels=in_channels,
            out_channels=out_channels,
            roi_size=roi_size,
        )
        n_val   = max(1, int(len(full_ds) * val_fraction))
        n_train = len(full_ds) - n_val
        train_sub, val_sub = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_sub if split == "train" else val_sub
        return DataLoader(
            chosen,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train"),
            pin_memory=torch.cuda.is_available(),
        )

    # ── Real data: random_split → per-split transform wrappers ──────────────
    raw_ds  = BraTSRawDataset(samples)
    n_val   = max(1, int(len(raw_ds) * val_fraction))
    n_train = len(raw_ds) - n_val
    train_raw, val_raw = random_split(
        raw_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    chosen_raw = train_raw if split == "train" else val_raw
    ds = BraTSTransformDataset(chosen_raw, split=split, roi_size=roi_size)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        drop_last=(split == "train"),
        pin_memory=torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Run ONE forward pass and return the scalar loss tensor WITH grad attached.

    Loss: DiceCELoss(sigmoid=True, to_onehot_y=False)
          — PyTorch/MONAI equivalent of BinaryDiceCELoss(from_logits=True,
            num_classes=3) from the original medicai script.

    Does NOT call loss.backward() or optimizer.step().
    The FL runtime is responsible for both.
    """
    device = next(model.parameters()).device

    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    logits = model(images)   # (B, out_channels, D, H, W)

    # Build and cache the loss function on first call
    if not hasattr(train_step, "_loss_fn"):
        if HAS_MONAI:
            # sigmoid=True     → applies sigmoid internally (from_logits equivalent)
            # to_onehot_y=False → label already multi-channel binary, not integer class
            train_step._loss_fn = _MonaiDiceCELoss(sigmoid=True, to_onehot_y=False)
        else:
            train_step._loss_fn = None

    if train_step._loss_fn is not None:
        loss = train_step._loss_fn(logits, labels)
    else:
        # Pure-PyTorch fallback: sigmoid BCE + soft Dice (mirrors BinaryDiceCELoss)
        bce   = F.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        inter  = (probs * labels).sum(dim=(2, 3, 4))
        union  = probs.sum(dim=(2, 3, 4)) + labels.sum(dim=(2, 3, 4))
        dice   = 1.0 - (2.0 * inter + smooth) / (union + smooth)
        loss   = bce + dice.mean()

    return loss
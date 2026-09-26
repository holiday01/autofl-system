import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


# ---------------------------------------------------------------------------
# Depthwise-separable convolution  (Keras SeparableConv2D equivalent)
# ---------------------------------------------------------------------------

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


# ---------------------------------------------------------------------------
# U-Net Xception-style model  (PyTorch port of the Keras get_model())
# ---------------------------------------------------------------------------

class XceptionUNet(nn.Module):
    """
    Fully-convolutional U-Net / Xception-style segmentation model.
    Faithfully mirrors the Keras architecture from the Oxford Pets example.

    Input  : (B, 3, H, W) float32 in [0, 1]
    Output : (B, num_classes, H, W) raw logits  (softmax is NOT applied;
              F.cross_entropy handles that, matching Keras
              sparse_categorical_crossentropy).
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        # ── Entry block: Conv → BN → ReLU  (stride-2 halves spatial dims) ──
        self.entry_conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.entry_bn   = nn.BatchNorm2d(32)

        # ── Downsampling: three blocks, filter widths [64, 128, 256] ────────
        # Each block: ReLU→SepConv→BN → ReLU→SepConv→BN → MaxPool
        #             + 1×1 stride-2 residual projection
        down_specs = [(32, 64), (64, 128), (128, 256)]
        self.down_sep1 = nn.ModuleList()
        self.down_bn1  = nn.ModuleList()
        self.down_sep2 = nn.ModuleList()
        self.down_bn2  = nn.ModuleList()
        self.down_pool = nn.ModuleList()
        self.down_res  = nn.ModuleList()
        for ic, oc in down_specs:
            self.down_sep1.append(SeparableConv2d(ic, oc, 3, padding=1))
            self.down_bn1.append(nn.BatchNorm2d(oc))
            self.down_sep2.append(SeparableConv2d(oc, oc, 3, padding=1))
            self.down_bn2.append(nn.BatchNorm2d(oc))
            self.down_pool.append(nn.MaxPool2d(3, stride=2, padding=1))
            self.down_res.append(nn.Conv2d(ic, oc, 1, stride=2, bias=False))

        # ── Upsampling: four blocks, filter widths [256, 128, 64, 32] ───────
        # Each block: ReLU→ConvTranspose→BN → ReLU→ConvTranspose→BN → Upsample
        #             + Upsample(prev) → 1×1 conv residual projection
        # ic = channels entering the block (== channels in `prev` at that point)
        # oc = channels produced by the block
        up_specs = [(256, 256), (256, 128), (128, 64), (64, 32)]
        self.up_tconv1 = nn.ModuleList()
        self.up_bn1    = nn.ModuleList()
        self.up_tconv2 = nn.ModuleList()
        self.up_bn2    = nn.ModuleList()
        self.up_up     = nn.ModuleList()
        self.up_res_up = nn.ModuleList()
        self.up_res    = nn.ModuleList()
        for ic, oc in up_specs:
            self.up_tconv1.append(nn.ConvTranspose2d(ic, oc, 3, padding=1, bias=False))
            self.up_bn1.append(nn.BatchNorm2d(oc))
            self.up_tconv2.append(nn.ConvTranspose2d(oc, oc, 3, padding=1, bias=False))
            self.up_bn2.append(nn.BatchNorm2d(oc))
            self.up_up.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.up_res_up.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.up_res.append(nn.Conv2d(ic, oc, 1, bias=False))

        # ── Per-pixel classification head ────────────────────────────────────
        self.out_conv = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry
        x    = F.relu(self.entry_bn(self.entry_conv(x)))
        prev = x

        # Downsampling
        for sep1, bn1, sep2, bn2, pool, res_proj in zip(
            self.down_sep1, self.down_bn1,
            self.down_sep2, self.down_bn2,
            self.down_pool, self.down_res,
        ):
            x    = bn1(sep1(F.relu(x)))
            x    = bn2(sep2(F.relu(x)))
            x    = pool(x)
            x    = x + res_proj(prev)
            prev = x

        # Upsampling
        for tconv1, bn1, tconv2, bn2, up, res_up, res_proj in zip(
            self.up_tconv1, self.up_bn1,
            self.up_tconv2, self.up_bn2,
            self.up_up, self.up_res_up, self.up_res,
        ):
            x    = bn1(tconv1(F.relu(x)))
            x    = bn2(tconv2(F.relu(x)))
            x    = up(x)
            x    = x + res_proj(res_up(prev))
            prev = x

        return self.out_conv(x)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class OxfordPetsDataset(Dataset):
    """Oxford-IIIT Pets segmentation dataset loaded from disk."""

    def __init__(self, input_img_paths, target_img_paths,
                 img_size=(160, 160)):
        n = min(len(input_img_paths), len(target_img_paths))
        self.input_img_paths  = input_img_paths[:n]
        self.target_img_paths = target_img_paths[:n]
        self.img_size         = img_size   # (H, W)

    def __len__(self):
        return len(self.input_img_paths)

    def __getitem__(self, idx):
        h, w = self.img_size

        # Image → float32 tensor (3, H, W) in [0, 1]
        img = Image.open(self.input_img_paths[idx]).convert("RGB")
        img = img.resize((w, h), Image.BILINEAR)
        img = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)

        # Trimap → int64 tensor (H, W), values in {0, 1, 2}
        # Original pixel values are 1/2/3; subtract 1 → 0/1/2.
        mask = Image.open(self.target_img_paths[idx]).convert("L")
        mask = mask.resize((w, h), Image.NEAREST)
        mask = torch.from_numpy(
            np.clip(np.array(mask, dtype=np.int64) - 1, 0, 2)
        )

        return img, mask


class _SyntheticSegDataset(Dataset):
    """Synthetic fallback; activated only when config['allow_synthetic_data']=True."""

    def __init__(self, size: int = 200, img_size=(160, 160), num_classes: int = 3):
        self.size        = size
        self.img_size    = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        h, w = self.img_size
        img  = torch.randn(3, h, w)
        mask = torch.randint(0, self.num_classes, (h, w))
        return img, mask


# ---------------------------------------------------------------------------
# FL API  ──  build_model / build_dataloader / train_step
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the XceptionUNet segmentation model.

    Recognised config keys (all optional):
        config["model_kwargs"]["num_classes"]  (default 3)
    """
    kwargs      = config.get("model_kwargs", {})
    num_classes = kwargs.get("num_classes", 3)
    return XceptionUNet(num_classes=num_classes)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Data layout expected under config['data_path']:
        images/                  – JPEG images   (*.jpg)
        annotations/trimaps/     – trimap PNGs   (*.png)

    Raises FileNotFoundError when the real dataset is absent **and**
    config['allow_synthetic_data'] is not True.  The synthetic fallback is
    never used silently.
    """
    batch_size  = config.get("local", {}).get("batch_size", 16)
    data_path   = config.get("data_path", ".")
    img_size    = tuple(config.get("model_kwargs", {}).get("img_size", [160, 160]))
    num_classes = config.get("model_kwargs", {}).get("num_classes", 3)

    input_dir  = os.path.join(data_path, "images")
    target_dir = os.path.join(data_path, "annotations", "trimaps")

    real_data_found = os.path.isdir(input_dir) and os.path.isdir(target_dir)

    if real_data_found:
        input_img_paths = sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(".jpg")
        )
        target_img_paths = sorted(
            os.path.join(target_dir, f)
            for f in os.listdir(target_dir)
            if f.lower().endswith(".png") and not f.startswith(".")
        )
        if len(input_img_paths) == 0:
            real_data_found = False

    if not real_data_found:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Oxford Pets dataset not found at '{data_path}'. "
                "Expected sub-directories 'images/' (*.jpg) and "
                "'annotations/trimaps/' (*.png). "
                "Set config['allow_synthetic_data'] = True to use synthetic "
                "data instead."
            )
        dataset = _SyntheticSegDataset(
            size=200, img_size=img_size, num_classes=num_classes
        )
    else:
        dataset = OxfordPetsDataset(input_img_paths, target_img_paths, img_size)

    val_size   = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1337),
    )

    chosen = train_ds if split == "train" else val_ds
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        drop_last=False,
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Single forward pass; returns the scalar loss WITH grad attached.

    The FL runtime calls loss.backward() and optimizer.step().
    This function must NOT do either.
    """
    device        = next(model.parameters()).device
    images, masks = batch
    images        = images.to(device)
    masks         = masks.to(device).long()

    logits = model(images)                    # (B, num_classes, H, W)
    loss   = F.cross_entropy(logits, masks)   # matches Keras sparse_categorical_crossentropy
    return loss
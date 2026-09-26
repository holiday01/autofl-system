"""
Auto-generated FL client module.
Original script: Keras Cats vs Dogs image classification from scratch
(fchollet, https://keras.io/examples/vision/image_classification_from_scratch/)

Exposes:
  build_model(config)                    -> nn.Module
  build_dataloader(config, split)        -> DataLoader
  train_step(model, batch, opt, config)  -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# ── Dataset ──────────────────────────────────────────────────────────────────

class CatsDogsDataset(Dataset):
    """
    Reads PetImages/{Cat,Dog}/*.jpg; skips images without a JFIF header.
    Falls back to 200 synthetic samples when the directory is absent (for testing).
    """

    def __init__(self, root: str, image_size: tuple = (180, 180), transform=None):
        self.transform = transform
        self.image_size = image_size
        self.samples = []

        for label_idx, class_name in enumerate(["Cat", "Dog"]):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                fpath = os.path.join(class_dir, fname)
                try:
                    with open(fpath, "rb") as f:
                        if b"JFIF" in f.read(10):
                            self.samples.append((fpath, label_idx))
                except (OSError, IOError):
                    pass

        self.synthetic = len(self.samples) == 0
        if self.synthetic:
            self.samples = [(None, i % 2) for i in range(200)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        if self.synthetic:
            x = torch.randn(3, *self.image_size)
        else:
            img = Image.open(fpath).convert("RGB")
            if self.transform:
                img = self.transform(img)
            x = img
        return x, torch.tensor(label, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class _SeparableConv2d(nn.Module):
    """Depthwise(k×k) followed by pointwise(1×1), matching Keras SeparableConv2D."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class _XceptionBlock(nn.Module):
    """
    One residual block from the mini-Xception architecture:
      ReLU → SepConv → BN → ReLU → SepConv → BN → MaxPool(stride=2)
    with a Conv2d(1×1, stride=2) skip connection.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.ReLU(),
            _SeparableConv2d(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            _SeparableConv2d(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False)

    def forward(self, x):
        return self.branch(x) + self.skip(x)


class MiniXception(nn.Module):
    """
    PyTorch port of the mini-Xception classifier from the Keras example.
    Expects float32 input in [0, 1] (standard torchvision ToTensor range).
    Binary classification (num_classes=2) emits a single logit;
    multi-class emits num_classes logits.
    """

    def __init__(self, num_classes: int = 2, image_size: tuple = (180, 180)):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            _XceptionBlock(128, 256),
            _XceptionBlock(256, 512),
            _XceptionBlock(512, 728),
        )
        self.final_conv = nn.Sequential(
            _SeparableConv2d(728, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)
        units = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(1024, units)

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# ── FL Interface ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 128))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory", True)

    image_size = tuple(config.get("image_size", [180, 180]))

    # RandomRotation(36): Keras RandomRotation(0.1) means ±10% of a full turn = ±36°
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(36),
        transforms.ToTensor(),  # PIL [0,255] -> float32 [0,1]
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    data_path = config.get("data_path", "PetImages")
    full_dataset = CatsDogsDataset(
        root=data_path,
        image_size=image_size,
        transform=train_transform if split == "train" else val_transform,
    )

    val_ratio = config.get("val_ratio", 0.2)
    n_val = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.get("seed", 1337)),
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

    # Single logit output → BCEWithLogitsLoss (binary); multi-logit → CrossEntropyLoss
    if outputs.shape[-1] == 1:
        loss = nn.BCEWithLogitsLoss()(outputs.squeeze(-1), targets.float())
    else:
        loss = nn.CrossEntropyLoss()(outputs, targets)
    return loss
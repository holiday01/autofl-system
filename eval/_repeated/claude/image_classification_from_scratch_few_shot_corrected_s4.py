"""
Auto-generated FL client module.
Original script: Keras image classification from scratch (Cats vs Dogs).

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ── Model ─────────────────────────────────────────────────────────────────────

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SeparableConv2d(in_channels, out_channels)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = SeparableConv2d(out_channels, out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(3, stride=2, padding=1)
        self.skip  = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.pool(x)
        return x + residual


class MiniXception(nn.Module):
    """Mini-Xception mirroring the Keras scratch image-classification architecture."""

    def __init__(self, num_classes=2, dropout=0.25):
        super().__init__()
        self.entry_conv = nn.Conv2d(3, 128, 3, stride=2, padding=1, bias=False)
        self.entry_bn   = nn.BatchNorm2d(128)
        self.block1     = ResidualBlock(128, 256)
        self.block2     = ResidualBlock(256, 512)
        self.block3     = ResidualBlock(512, 728)
        self.top_conv   = SeparableConv2d(728, 1024)
        self.top_bn     = nn.BatchNorm2d(1024)
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(dropout)
        units = 1 if num_classes == 2 else num_classes
        self.head = nn.Linear(1024, units)

    def forward(self, x):
        x = F.relu(self.entry_bn(self.entry_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.top_bn(self.top_conv(x)))
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)


# ── FL Interface ───────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return MiniXception(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 128))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory", True)

    image_size = tuple(config.get("image_size", (180, 180)))
    data_path  = config.get("data_path", "PetImages")

    aug_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=36),  # 0.1 * 360, matching original
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    chosen_transform = aug_transform if split == "train" else val_transform

    if os.path.isdir(data_path):
        dataset = datasets.ImageFolder(root=data_path, transform=chosen_transform)
    else:
        # synthetic fallback for testing without data on disk
        from torch.utils.data import TensorDataset
        n = config.get("synthetic_n", 400)
        X = torch.rand(n, 3, *image_size)
        y = torch.randint(0, 2, (n,))
        dataset = TensorDataset(X, y)

    val_ratio = config.get("val_ratio", 0.2)
    n_val     = max(1, int(len(dataset) * val_ratio))
    n_train   = len(dataset) - n_val

    gen     = torch.Generator().manual_seed(config.get("seed", 1337))
    indices = torch.randperm(len(dataset), generator=gen).tolist()
    split_indices = indices[:n_train] if split == "train" else indices[n_train:]
    ds = Subset(dataset, split_indices)

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
    num_classes = config.get("model_kwargs", {}).get("num_classes", 2)
    if num_classes == 2:
        # BCEWithLogitsLoss mirrors Keras BinaryCrossentropy(from_logits=True)
        outputs = outputs.squeeze(1)
        loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
    else:
        loss = nn.CrossEntropyLoss()(outputs, targets)
    return loss
"""
Auto-generated FL client module.
Original script: image classification from scratch (Cats vs Dogs / Keras mini-Xception).

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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class SeparableConv2d(nn.Module):
    """Depthwise-separable convolution: depthwise + pointwise, matching Keras SeparableConv2D."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class MiniXception(nn.Module):
    """
    PyTorch port of the mini-Xception architecture from the Keras cats-vs-dogs example.
    Rescaling (1/255) is omitted here; ToTensor() in the dataloader handles it.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        self.projections = nn.ModuleList()
        in_ch = 128
        for out_ch in [256, 512, 728]:
            self.blocks.append(nn.Sequential(
                nn.ReLU(inplace=True),
                SeparableConv2d(in_ch, out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                SeparableConv2d(out_ch, out_ch),
                nn.BatchNorm2d(out_ch),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ))
            self.projections.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False)
            )
            in_ch = out_ch
        self.top = nn.Sequential(
            SeparableConv2d(728, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        units = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Linear(1024, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        for block, proj in zip(self.blocks, self.projections):
            x = block(x) + proj(x)
        x = self.top(x)
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

    image_size = tuple(config.get("image_size", (180, 180)))
    data_path  = config.get("data_path", "PetImages")
    val_ratio  = config.get("val_ratio", 0.2)
    seed       = config.get("seed", 1337)

    train_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # Build two dataset views so each split gets its own transform without sharing state.
    full_train = datasets.ImageFolder(root=data_path, transform=train_tf)
    full_val   = datasets.ImageFolder(root=data_path, transform=val_tf)

    n_total = len(full_train)
    n_val   = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
    train_ds = Subset(full_train, indices[:n_train])
    val_ds   = Subset(full_val,   indices[n_train:])

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
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
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

    outputs = model(inputs)  # shape: (B, 1) for binary, (B, C) for multiclass

    num_classes = config.get("model_kwargs", {}).get("num_classes", 2)
    if num_classes == 2:
        # BCEWithLogitsLoss mirrors Keras BinaryCrossentropy(from_logits=True)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs.squeeze(1), targets.float())
    else:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
    return loss
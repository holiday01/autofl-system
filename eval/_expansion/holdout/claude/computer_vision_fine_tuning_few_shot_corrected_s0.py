"""
Auto-generated FL client module.
Original script: computer_vision_fine_tuning.py

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT:
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.models import resnet50
from pathlib import Path


DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"


class TransferLearningClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet50", freeze_backbone: bool = True):
        super().__init__()
        if backbone == "resnet50":
            base = resnet50(weights="DEFAULT")
        else:
            import torchvision.models as tvm
            base = getattr(tvm, backbone)(weights="DEFAULT")

        _layers = list(base.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.fc(x)


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return TransferLearningClassifier(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 8))
    num_workers = local.get("num_workers", config.get("num_workers", 0))
    pin_memory  = local.get("pin_memory", True)

    dl_path = config.get("data_path", "data")
    data_path = Path(dl_path) / "cats_and_dogs_filtered"

    if config.get("download_data", False) and not data_path.exists():
        download_and_extract_archive(url=DATA_URL, download_root=dl_path, remove_finished=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    if split == "train":
        train_dir = data_path / "train"
        if train_dir.exists():
            dataset = ImageFolder(root=str(train_dir), transform=train_transform)
        else:
            dataset = _synthetic_dataset(train_transform, n=200)
        shuffle = True
        transform_used = train_transform
    else:
        val_dir = data_path / "validation"
        if val_dir.exists():
            dataset = ImageFolder(root=str(val_dir), transform=val_transform)
        else:
            dataset = _synthetic_dataset(val_transform, n=50)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def _synthetic_dataset(transform, n: int = 200):
    """Fallback in-memory dataset when real data is unavailable."""
    from torch.utils.data import TensorDataset
    imgs = torch.randn(n, 3, 224, 224)
    labels = torch.randint(0, 2, (n,))
    return TensorDataset(imgs, labels)


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

    logits = model(inputs)
    y_true = targets.view(-1, 1).to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, y_true)
    return loss
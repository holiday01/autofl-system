"""
Auto-generated FL client module.
Original script: backbone_image_classifier.py

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


class Backbone(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


class MNISTDataset(Dataset):
    """Wraps torchvision MNIST; raises on missing data so callers can decide fallback."""

    def __init__(self, root: str, train: bool = True):
        from torchvision import transforms
        from torchvision.datasets import MNIST
        self._ds = MNIST(root, train=train, download=False, transform=transforms.ToTensor())

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


class SyntheticMNISTDataset(Dataset):
    def __init__(self, n: int = 200):
        self.X = torch.randn(n, 1, 28, 28)
        self.y = torch.randint(0, 10, (n,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return Backbone(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    data_path = config.get("data_path", ".")
    allow_synthetic = config.get("allow_synthetic_data", False)
    use_train_split = split in ("train", "val")

    try:
        full_dataset = MNISTDataset(root=data_path, train=use_train_split)
    except Exception:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"MNIST data not found at '{data_path}' and allow_synthetic_data is False."
            )
        full_dataset = SyntheticMNISTDataset(n=config.get("synthetic_n", 200))

    if split == "test":
        ds = full_dataset
    else:
        val_ratio = config.get("val_ratio", 5000 / 60000)
        n_val = max(1, int(len(full_dataset) * val_ratio))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
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
    loss = F.cross_entropy(outputs, targets)
    return loss
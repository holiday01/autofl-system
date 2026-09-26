"""
Auto-generated FL client module (based on new script).
Original script: backbone_image_classifier.py (from Lightning AI examples)

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
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


# ── Model Definition ────────────────────────────────────────────────────

class Backbone(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        # The original LitClassifier flattened x before passing to backbone.
        # This backbone expects an already flattened input.
        # Flattening will be handled in FLLitClassifier's forward.
        x = F.relu(self.l1(x))
        return F.relu(self.l2(x))


class FLLitClassifier(nn.Module):
    """
    Adapted from LightningModule to a standard nn.Module for FL.
    It wraps the Backbone and handles the initial flattening of the input.
    """
    def __init__(self, backbone_hidden_dim: int = 128):
        super().__init__()
        self.backbone = Backbone(hidden_dim=backbone_hidden_dim)

    def forward(self, x):
        # Flatten the image from (batch, 1, 28, 28) to (batch, 784)
        x = x.view(x.size(0), -1)
        return self.backbone(x)


# ── Data Definition Helpers ─────────────────────────────────────────────

_DEFAULT_TRANSFORM = transforms.ToTensor()


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the model based on the provided configuration.
    """
    model_kwargs = config.get("model_kwargs", {})
    # Extract 'hidden_dim' which corresponds to 'backbone_hidden_dim' in FLLitClassifier
    backbone_hidden_dim = model_kwargs.get("hidden_dim", 128)
    return FLLitClassifier(backbone_hidden_dim=backbone_hidden_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a DataLoader for the specified split (train or val).
    """
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)
    seed = config.get("seed", 42)

    data_path = config.get("data_path", ".")
    os.makedirs(data_path, exist_ok=True) # Ensure data_path exists for download

    # Load the full MNIST training dataset (60,000 samples)
    full_mnist_train = MNIST(root=data_path, train=True, download=True, transform=_DEFAULT_TRANSFORM)
    # Load the MNIST test dataset (10,000 samples)
    mnist_test = MNIST(root=data_path, train=False, download=True, transform=_DEFAULT_TRANSFORM)

    # Replicate the split logic from the original MyDataModule
    # Original: 55000 train, 5000 val from the 60000 sample train set
    train_size_orig = 55000
    val_size_orig = 5000

    # Adjust sizes proportionally if the total dataset length is not 60000 (e.g., for custom subsets)
    if len(full_mnist_train) != (train_size_orig + val_size_orig):
        total_len = len(full_mnist_train)
        val_ratio = val_size_orig / (train_size_orig + val_size_orig)
        new_val_size = int(total_len * val_ratio)
        new_train_size = total_len - new_val_size
        train_size = new_train_size
        val_size = new_val_size
    else:
        train_size = train_size_orig
        val_size = val_size_orig

    train_ds, val_ds = random_split(
        full_mnist_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    if split == "train":
        ds = train_ds
    elif split == "val":
        ds = val_ds
    elif split == "test": # FL clients usually only have train/val, but test might be configured.
        ds = mnist_test
    else:
        raise ValueError(f"Unsupported split: {split}. Expected 'train', 'val', or 'test'.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"), # Only shuffle training data
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

    # Handle various batch types, similar to the example
    if isinstance(batch, (list, tuple)):
        # Assuming (inputs, targets) tuple structure from DataLoader
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        # Attempt to find common keys for inputs and targets
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
        if inputs is None or targets is None:
            raise ValueError("Could not find 'inputs' or 'targets' in batch dictionary.")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # Move inputs and targets to the correct device
    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    return loss
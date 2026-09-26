"""
Auto-generated FL client module.
Original script: backbone_image_classifier.py

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
from torch.utils.data import DataLoader, random_split
from torchvision import transforms # Assuming torchvision is available
from lightning.pytorch.demos.mnist_datamodule import MNIST # This dependency is needed


# --- Model Definitions (Adapted from original script) ---

class Backbone(torch.nn.Module):
    """
    Backbone model for MNIST classification.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the image
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the model for FL training.
    The `LitClassifier` from the original script is stripped down,
    and only its `Backbone` component is returned, as the FL runtime
    handles the training loop and optimization.
    """
    kwargs = config.get("model_kwargs", {})
    hidden_dim = kwargs.get("hidden_dim", 128)
    return Backbone(hidden_dim=hidden_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a DataLoader for a specific data split (train, val, or test).
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 32))
    # Default num_workers to 0 to match DataLoader default if not specified
    num_workers = local.get("num_workers", config.get("num_workers", 0))
    pin_memory = local.get("pin_memory", True)

    # Resolve data path, default to "Datasets" in current directory
    # Original script uses relative path: path.join(path.dirname(__file__), "..", "..", "Datasets")
    data_path = config.get("data_path", os.path.join(".", "Datasets"))
    seed = config.get("seed", 42) # Seed for random_split

    transform = transforms.ToTensor()

    # Create full datasets as in MyDataModule
    full_train_dataset = MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = MNIST(data_path, train=False, download=True, transform=transform)

    # Perform the train/val split as in MyDataModule
    train_len = 55000
    val_len = 5000
    
    # Using torch.Generator().manual_seed for reproducibility of split
    mnist_train_ds, mnist_val_ds = random_split(
        full_train_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    if split == "train":
        ds = mnist_train_ds
        shuffle = True
    elif split == "val":
        ds = mnist_val_ds
        shuffle = False
    elif split == "test":
        ds = test_dataset
        shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,  # Not used here as per contract
    config: dict,
) -> torch.Tensor:
    """
    Performs one forward pass and returns the raw loss tensor.
    The FL runtime handles `loss.backward()`, `optimizer.step()`, and metric extraction.
    """
    device = next(model.parameters()).device

    # Move batch to device and extract inputs, targets
    if isinstance(batch, (list, tuple)):
        # Original script uses x, y = batch
        inputs, targets = batch[0].to(device), batch[1].to(device)
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # Perform forward pass
    outputs = model(inputs)

    # Calculate loss (CrossEntropyLoss for classification)
    loss = F.cross_entropy(outputs, targets)
    return loss
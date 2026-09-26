"""
Auto-generated FL client module.
Original script: backbone_image_classifier.py (from PyTorch Lightning examples)

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

# Using torchvision.datasets.MNIST directly as it's the underlying dataset
from torchvision import transforms
from torchvision.datasets import MNIST


class Backbone(nn.Module):
    """
    The core neural network module from the original script.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = F.relu(self.l1(x))
        return F.relu(self.l2(x))


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the model.
    The original script's LitClassifier wraps a Backbone.
    For FL, we typically want the core nn.Module that performs the forward pass.
    Here, the Backbone is that core module.
    """
    kwargs = config.get("model_kwargs", {})
    # The 'hidden_dim' parameter for Backbone can be passed via model_kwargs.
    return Backbone(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns the DataLoader for the specified split.
    Replicates the data loading and splitting logic from MyDataModule.
    """
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 0))
    pin_memory  = local.get("pin_memory", True)
    data_path   = config.get("data_path", "./data") # Default data path

    # Ensure data_path exists for MNIST download
    os.makedirs(data_path, exist_ok=True)

    transform = transforms.ToTensor()

    # Load the full MNIST training dataset
    full_train_dataset = MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = MNIST(data_path, train=False, download=True, transform=transform)

    # Split the training dataset into train and validation
    # Use the seed from config for reproducibility
    seed = config.get("seed", 42)
    generator = torch.Generator().manual_seed(seed)
    
    # Original split was 55000 for train, 5000 for val.
    # We can make this configurable via val_ratio if needed, but for direct conversion,
    # we'll use the fixed sizes.
    n_train_orig = 55000
    n_val_orig = 5000
    
    # Adjust split sizes if the total dataset size doesn't match or if val_ratio is provided
    total_train_size = len(full_train_dataset)
    if total_train_size != (n_train_orig + n_val_orig):
        val_ratio = config.get("val_ratio", n_val_orig / (n_train_orig + n_val_orig))
        n_val = max(1, int(total_train_size * val_ratio))
        n_train = total_train_size - n_val
    else:
        n_train, n_val = n_train_orig, n_val_orig
        
    mnist_train, mnist_val = random_split(
        full_train_dataset, [n_train, n_val], generator=generator
    )

    if split == "train":
        ds = mnist_train
        shuffle = True
    elif split == "val":
        ds = mnist_val
        shuffle = False
    elif split == "test":
        ds = mnist_test
        shuffle = False
    else:
        raise ValueError(f"Unsupported split: {split}. Must be 'train', 'val', or 'test'.")

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
    optimizer, # Not used in this function, but part of the contract
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    
    # The original script's training_step expects (x, y)
    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0], batch[1]
    else:
        # Handle potential dict batches if they were ever introduced, though MNIST is usually tuple
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    return loss
"""
Auto-generated FL client module.
Original script: (provided by user)

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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # The input dimension to fc1 (9216) is specific to MNIST's 28x28 input
        # after the defined convolutional and pooling layers.
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the model.
    The Net class does not take any initialization arguments, so no kwargs are needed.
    """
    return Net()


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns the DataLoader for the specified split.
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 0))
    pin_memory = local.get("pin_memory", True)
    # Default data_path to '../data' as in the original script, but make it configurable
    data_path = config.get("data_path", "../data")
    seed = config.get("seed", 1) # For reproducibility if random_split were used

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if split == "train":
        is_train = True
        shuffle = True
        dataset = datasets.MNIST(data_path, train=is_train, download=True, transform=transform)
    elif split == "test" or split == "val":
        # MNIST has a dedicated test set, so 'val' can map to 'test'
        is_train = False
        shuffle = False # No need to shuffle test/validation data
        # Use test_batch_size if available in config, otherwise fallback to batch_size
        batch_size = local.get("test_batch_size", config.get("test_batch_size", batch_size))
        dataset = datasets.MNIST(data_path, train=is_train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported split: {split}. Expected 'train', 'test', or 'val'.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer, # Optimizer is passed but not used in this function as per contract
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    # The original script's train function expects (data, target)
    if isinstance(batch, (list, tuple)):
        data, target = batch
    elif isinstance(batch, dict):
        # Handle potential dict batches if the DataLoader was customized
        data = batch.get("input", batch.get("x", batch.get("image")))
        target = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    data, target = data.to(device), target.to(device)

    output = model(data)
    # Use F.nll_loss as in the original script
    loss = F.nll_loss(output, target)

    return loss
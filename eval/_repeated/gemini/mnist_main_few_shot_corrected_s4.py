"""
Auto-generated FL client module.
Original script: [refer to the script you provided]

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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
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
    Builds and returns the neural network model.
    The original Net class does not take parameters, so model_kwargs is not used here.
    """
    return Net()


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a DataLoader for the specified split (train or test).
    """
    local = config.get("local", {})

    # Determine batch size based on split, falling back to general config or defaults
    if split == "train":
        batch_size = local.get("batch_size", config.get("batch_size", 64))
    elif split == "test":
        batch_size = local.get("test_batch_size", config.get("test_batch_size", 1000))
    else:
        raise ValueError(f"Unsupported split: {split}. Expected 'train' or 'test'.")

    num_workers = local.get("num_workers", config.get("num_workers", 1))
    pin_memory = local.get("pin_memory", True)

    data_path = config.get("data_path", "./data") # Default data directory

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if split == "train":
        dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    elif split == "test":
        dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Use a generator for reproducibility if a seed is provided in the config
    generator = None
    seed = config.get("seed")
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),  # Shuffle only for training data
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        generator=generator,
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer, # The optimizer is passed but its step/zero_grad methods are NOT called here
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    # Ensure model and data are on the same device
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0].to(device), batch[1].to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # Perform a forward pass
    outputs = model(inputs)

    # Calculate the loss
    loss = F.nll_loss(outputs, targets)

    return loss
"""
Auto-generated FL client module.
Original script: [PATH_TO_YOUR_ORIGINAL_SCRIPT.py]

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


# ── Model Definition ────────────────────────────────────────────────────────

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
    # The Net model in the original script does not take __init__ arguments,
    # but we keep model_kwargs for flexibility if the model changes.
    kwargs = config.get("model_kwargs", {})
    return Net(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    
    # Batch size can be specified per split or globally
    batch_size_key = "batch_size" if split == "train" else "test_batch_size"
    batch_size_default = 64 if split == "train" else 1000 # Matching original script defaults
    batch_size = local.get(batch_size_key, config.get(batch_size_key, batch_size_default))

    # num_workers and pin_memory from config, matching exemplar's style
    # Original script used num_workers=1 for accelerated case, 0 for non-accel default
    num_workers = local.get("num_workers", config.get("num_workers", 0))
    pin_memory  = local.get("pin_memory", True)

    # Data path for MNIST dataset
    data_path = config.get("data_path", "../data")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    is_train_split = (split == "train")
    dataset = datasets.MNIST(
        root=data_path,
        train=is_train_split,
        download=True, # Allow downloading if dataset is not present locally
        transform=transform
    )
    
    # Shuffle only training data, as is standard practice
    shuffle_data = is_train_split

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer, # This parameter is part of the contract but not used within train_step
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    
    # Extract data and target from the batch and move to device
    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0].to(device), batch[1].to(device)
    elif isinstance(batch, dict):
        # Handle dict batches if they appear, though MNIST typically provides tuple batches
        inputs  = batch.get("input", batch.get("x", batch.get("image"))).to(device)
        targets = batch.get("label", batch.get("y", batch.get("target"))).to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    loss = F.nll_loss(outputs, targets) # The original script used F.nll_loss

    return loss
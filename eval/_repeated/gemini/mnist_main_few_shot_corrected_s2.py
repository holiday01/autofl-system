import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


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
    # The Net model does not take any arguments in its constructor,
    # so we can simply instantiate it.
    # If it needed arguments, we would use config.get("model_kwargs", {})
    return Net()


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 1))
    pin_memory = local.get("pin_memory", True)
    
    # Original script uses "../data", making this configurable via data_path
    data_path = config.get("data_path", "../data")
    seed = config.get("seed", 42) # Using the example's default seed

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if split == "train" or split == "val":
        # For 'train' and 'val' splits, we load the full MNIST training set
        # and then split it according to the val_ratio, following the example pattern.
        full_train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        
        val_ratio = config.get("val_ratio", 0.1)
        n_val = max(1, int(len(full_train_dataset) * val_ratio))
        n_train = len(full_train_dataset) - n_val
        
        # Use a generator for reproducible split
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(full_train_dataset, [n_train, n_val], generator=generator)
        
        ds = train_ds if split == "train" else val_ds
        shuffle_data = (split == "train")
    elif split == "test":
        # For a 'test' split, we use the official MNIST test set.
        ds = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        shuffle_data = False # Typically no shuffle for evaluation
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer, # Not used here as per contract, but part of signature
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    
    # MNIST DataLoader yields (data, target) tuples
    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0].to(device), batch[1].to(device)
    elif isinstance(batch, dict):
        # Fallback for dict batches, consistent with exemplar, though not typical for MNIST
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    
    # The original script uses F.nll_loss directly because Net's forward
    # already applies F.log_softmax.
    loss = F.nll_loss(outputs, targets)
    return loss
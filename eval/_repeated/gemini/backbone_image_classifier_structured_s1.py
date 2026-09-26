import os
from os import path
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms

# Conditional import for torchvision.datasets.MNIST
_HAS_TORCHVISION_MNIST = False
try:
    from torchvision.datasets import MNIST
    _HAS_TORCHVISION_MNIST = True
except ImportError:
    pass # torchvision not available


class Backbone(torch.nn.Module):
    """
    >>> Backbone()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Backbone(
      (l1): Linear(...)
      (l2): Linear(...)
    )
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


class LitClassifier(torch.nn.Module):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(self, backbone: Optional[Backbone] = None, learning_rate: float = 0.0001):
        super().__init__()
        if backbone is None:
            backbone = Backbone()
        self.backbone = backbone
        # Store learning_rate, although it's typically managed by the FL runtime's optimizer
        self.learning_rate = learning_rate 

    def forward(self, x):
        # use forward for inference/predictions
        return self.backbone(x)


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    
    # Extract arguments for the Backbone
    backbone_kwargs = model_kwargs.get("backbone_kwargs", {})
    hidden_dim = backbone_kwargs.get("hidden_dim", 128)
    backbone = Backbone(hidden_dim=hidden_dim)

    # Extract arguments for the LitClassifier
    classifier_kwargs = model_kwargs.get("classifier_kwargs", {})
    learning_rate = classifier_kwargs.get("learning_rate", 0.0001)

    model = LitClassifier(backbone=backbone, learning_rate=learning_rate)
    return model


class SyntheticMNISTDataset(Dataset):
    """
    A synthetic dataset to mimic MNIST data for fallback scenarios.
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        # MNIST images are 1 channel, 28x28 pixels
        self.data = torch.randn(num_samples, 1, 28, 28)
        # MNIST labels are integers from 0-9
        self.targets = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    if split not in ["train", "val"]:
        raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'.")

    dataset = None
    if _HAS_TORCHVISION_MNIST:
        try:
            # The original script uses MNIST(train=True) and then splits it.
            # We follow this approach for consistency and to provide train/val splits
            # from the standard 'training' part of MNIST.
            full_train_dataset = MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
            
            total_size = len(full_train_dataset)
            # Replicate the original random_split sizes [55000, 5000] if dataset is standard 60k samples.
            # Otherwise, apply a proportional split.
            if total_size == 60000:
                train_size = 55000
                val_size = 5000
            else: 
                train_size = int(0.91666 * total_size) # Approximately 55000/60000
                val_size = total_size - train_size
            
            # Use a fixed generator seed for reproducibility, as in the original script
            generator = torch.Generator().manual_seed(42)
            train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

            if split == "train":
                dataset = train_subset
            elif split == "val":
                dataset = val_subset

        except Exception as e:
            # If real data loading fails
            if not allow_synthetic_data:
                raise FileNotFoundError(
                    f"Real dataset unavailable at '{data_path}' and 'allow_synthetic_data' is False. "
                    f"Original error: {e}"
                ) from e
            print(f"Warning: Could not load real MNIST dataset from '{data_path}' due to: {e}. "
                  "Proceeding with synthetic data as 'allow_synthetic_data' is True.")
    else: # torchvision.datasets.MNIST is not available
        if not allow_synthetic_data:
            raise ImportError(
                "torchvision.datasets.MNIST is not available and 'allow_synthetic_data' is False. "
                "Cannot build dataloader without a dataset source."
            )
        print("Warning: torchvision.datasets.MNIST is not available. Proceeding with synthetic data.")

    # If dataset is still None at this point, it means we must use synthetic data
    # (because allow_synthetic_data was True and real data either failed to load or torchvision was absent).
    if dataset is None:
        print(f"Generating synthetic data for split: {split}")
        num_synthetic_samples = 1000 # Example fixed size for synthetic data
        if split == "train":
            dataset = SyntheticMNISTDataset(num_samples=int(num_synthetic_samples * 0.8))
        elif split == "val":
            dataset = SyntheticMNISTDataset(num_samples=int(num_synthetic_samples * 0.2))

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    """
    # Move tensors to the device of the model parameters
    device = next(model.parameters()).device
    x, y = batch[0].to(device), batch[1].to(device)

    # Forward pass
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)

    # Return the loss tensor with grad attached.
    # Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    return loss
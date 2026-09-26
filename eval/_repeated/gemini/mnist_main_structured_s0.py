import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import numpy as np
import os


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


class SyntheticMNIST(Dataset):
    def __init__(self, num_samples=1000, transform=None):
        self.num_samples = num_samples
        # Generate unnormalized, uint8 data to mimic raw MNIST (28x28 grayscale)
        self.raw_images = torch.randint(0, 256, (num_samples, 28, 28), dtype=torch.uint8)
        self.labels = torch.randint(0, 10, (num_samples,), dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Convert torch.uint8 tensor to numpy array, then to PIL Image
        image_np = self.raw_images[idx].numpy() # HxW numpy array
        image_pil = Image.fromarray(image_np, mode='L') # Grayscale PIL image

        if self.transform:
            image = self.transform(image_pil)
        else:
            # Apply default MNIST transforms if none provided
            image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])(image_pil)

        label = self.labels[idx]
        return image, label


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    return Net(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    seed = config.get("seed", 42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = None
    try:
        # Attempt to load the real MNIST training dataset
        full_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    except Exception as e:
        if allow_synthetic_data:
            print(f"Warning: Could not load real MNIST data from {data_path}. Using synthetic data. Error: {e}")
            full_dataset = SyntheticMNIST(num_samples=1000, transform=transform)
        else:
            raise FileNotFoundError(
                f"Real MNIST dataset not found at '{data_path}' and 'allow_synthetic_data' is False. "
                f"Please ensure the data path is correct or enable synthetic data generation. Error: {e}"
            ) from e

    # Split the dataset into training and validation parts
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Ensure reproducibility of the split
    g_cpu = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g_cpu)

    if split == "train":
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    elif split == "val":
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")


def train_step(model: torch.nn.Module, batch, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass, calculate loss, and return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step().
    """
    model.train() # Set model to training mode
    device = next(model.parameters()).device # Get the device of the model

    data, target = batch
    data, target = data.to(device), target.to(device)

    # Note: optimizer.zero_grad() is usually called before the forward pass
    # but the FL runtime might manage optimizer steps including zero_grad.
    # We follow the prompt to only do the forward pass and loss calculation.
    # If the FL runtime expects client to zero_grad, it will instruct or wrap this.

    output = model(data)
    loss = F.nll_loss(output, target)

    return loss
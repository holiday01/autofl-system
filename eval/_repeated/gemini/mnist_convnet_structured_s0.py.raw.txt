import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import os

# Original model / data parameters from the Keras script
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)  # Keras expects (height, width, channels)

# --- PyTorch Model Equivalent to Keras Sequential ---
class SimpleConvNet(nn.Module):
    """
    A PyTorch equivalent of the Keras Sequential model from the original script.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, input_channels: int = INPUT_SHAPE[2]):
        super().__init__()
        # Keras input_shape is (height, width, channels).
        # PyTorch expects (channels, height, width).
        # We define conv layers with `in_channels` based on the input_channels.

        # Original Keras layers:
        # layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=32,
                               kernel_size=(3, 3))
        self.relu1 = nn.ReLU()

        # layers.MaxPooling2D(pool_size=(2, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(3, 3))
        self.relu2 = nn.ReLU()

        # layers.MaxPooling2D(pool_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # layers.Flatten()
        self.flatten = nn.Flatten()

        # layers.Dropout(0.5)
        self.dropout = nn.Dropout(0.5)

        # layers.Dense(num_classes, activation="softmax")
        # To calculate in_features for the Linear layer:
        # Input (1, 28, 28)
        # Conv1: (C=1, H=28, W=28) -> (32, 26, 26)
        # Pool1: (32, 26, 26) -> (32, 13, 13)
        # Conv2: (32, 13, 13) -> (64, 11, 11)
        # Pool2: (64, 11, 11) -> (64, 5, 5) (after integer division for pooling)
        # Flatten: 64 * 5 * 5 = 1600
        self.fc = nn.Linear(in_features=64 * 5 * 5, out_features=num_classes)
        # Note: PyTorch's CrossEntropyLoss expects raw logits, so no softmax here.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is expected to be in PyTorch format: (batch_size, channels, height, width)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# --- Custom Dataset for MNIST-like data ---
class MNISTDataset(Dataset):
    """
    A simple PyTorch Dataset for MNIST-like images and integer labels.
    """
    def __init__(self, data: np.ndarray, targets: np.ndarray):
        # Data expected as (N, C, H, W) float32 for images, (N,) long for labels
        self.data = torch.from_numpy(data)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        return image, label

# --- FL Client Module Functions ---

def build_model(config: dict) -> nn.Module:
    """
    Instantiates and returns the PyTorch model.
    """
    model_kwargs = config.get("model_kwargs", {})
    num_classes = model_kwargs.get("num_classes", NUM_CLASSES)
    input_channels = model_kwargs.get("input_shape", INPUT_SHAPE)[2] # Extract channels from (H, W, C)
    return SimpleConvNet(num_classes=num_classes, input_channels=input_channels)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Returns a DataLoader for the requested split ("train" or "val").
    Handles data loading, preprocessing, and synthetic data fallback.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    val_split_ratio = config.get("val_split_ratio", 0.1)  # Keras default validation_split

    x_train_full = None
    y_train_full = None

    try:
        # Attempt to load real MNIST data using torchvision for robustness
        # This implicitly handles downloading if data_path is writeable and not present.
        from torchvision import datasets, transforms

        # Ensures the datasets are downloaded
        _ = datasets.MNIST(root=data_path, train=True, download=True)
        _ = datasets.MNIST(root=data_path, train=False, download=True)

        # Load data directly from torchvision datasets as tensors
        train_dataset_torch = datasets.MNIST(root=data_path, train=True, download=False)
        # Convert to numpy and then apply original script's preprocessing
        # to ensure exact replication of original logic if desired.
        x_train_full = train_dataset_torch.data.numpy()
        y_train_full = train_dataset_torch.targets.numpy()

        # Original script also loads x_test, y_test but for client's 'train' and 'val'
        # splits, we only need the client's 'training' dataset.
        # x_test, y_test will be used by the server for evaluation, not by this client's build_dataloader.

    except Exception as e:
        if allow_synthetic_data:
            print(f"WARNING: Could not load real data from {data_path}. Using synthetic data. Error: {e}")
            x_train_full = None # Mark as failed to load real data
            y_train_full = None
        else:
            raise FileNotFoundError(
                f"Could not load real data from {data_path} (e.g., MNIST). "
                "Set 'allow_synthetic_data' to True in the config "
                "to use synthetic data instead, or ensure torchvision/MNIST data is available."
            ) from e

    if x_train_full is None: # Synthetic data generation path
        num_samples_full = 60000 # Mimic MNIST train set size
        # Synthetic data with PyTorch-expected dimensions (N, C, H, W) and range [0, 1]
        x_train_full = np.random.rand(num_samples_full, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]).astype(np.float32)
        y_train_full = np.random.randint(0, NUM_CLASSES, num_samples_full, dtype=np.int64)
        # No need for /255 as it's already [0,1], no expand_dims needed as it's (H,W,C) already for synthetic

    else: # Real data preprocessing path
        # Scale images to the [0, 1] range
        x_train_full = x_train_full.astype("float32") / 255.0
        # Make sure images have shape (28, 28, 1) - Keras format
        x_train_full = np.expand_dims(x_train_full, -1) # (N, H, W) -> (N, H, W, C)
        # y_train_full is already integers, which PyTorch's CrossEntropyLoss expects.
        # Original Keras script used `to_categorical` which creates one-hot labels,
        # but PyTorch's CrossEntropyLoss expects class indices, so we keep y_train_full as integers.

    # Convert to PyTorch tensors and permute for PyTorch's (N, C, H, W) format
    x_train_full_pt = torch.from_numpy(x_train_full).permute(0, 3, 1, 2).contiguous() # (N, H, W, C) -> (N, C, H, W)
    y_train_full_pt = torch.from_numpy(y_train_full).long() # Ensure labels are LongTensor

    full_client_dataset = MNISTDataset(data=x_train_full_pt, targets=y_train_full_pt)

    # Split the client's full dataset into training and validation parts
    train_size = int((1 - val_split_ratio) * len(full_client_dataset))
    val_size = len(full_client_dataset) - train_size

    # Ensure reproducibility of random_split for consistent train/val splits across clients/runs
    # A fixed generator might be desirable in FL for client splits, but for client-side
    # train/val split, a fresh one is fine if not required otherwise.
    generator = torch.Generator().manual_seed(42 + config.get("client_id", 0)) if "client_id" in config else None
    train_dataset, val_dataset = random_split(full_client_dataset, [train_size, val_size], generator=generator)

    if split == "train":
        dataset_to_load = train_dataset
    elif split == "val":
        dataset_to_load = val_dataset
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'.")

    return DataLoader(dataset_to_load, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Runs ONE forward pass and returns the loss tensor with gradient attached.
    Does NOT call loss.backward() or optimizer.step().
    """
    images, labels = batch

    # Determine device of model parameters
    device = next(model.parameters()).device

    # Move tensors to the appropriate device
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)

    # Calculate loss
    # Keras used 'categorical_crossentropy' with one-hot labels.
    # PyTorch's `nn.CrossEntropyLoss` is suitable for multi-class classification,
    # expecting raw logits from the model and integer class indices for labels.
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    return loss
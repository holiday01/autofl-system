import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os

# Model / data parameters derived from the original script
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1) # Keras format (height, width, channels)

# --------------------------------------------------------------------------------------------------
# 1. build_model
# --------------------------------------------------------------------------------------------------

class KerasLikeConvNet(nn.Module):
    """
    Equivalent PyTorch model for the Keras Simple MNIST convnet.
    Preserves the original architecture and activation functions.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Input (1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding='valid'), # Output (32, 26, 26)
            nn.ReLU(),
            nn.MaxPool2d(pool_size=(2, 2)),                       # Output (32, 13, 13)
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='valid'),# Output (64, 11, 11)
            nn.ReLU(),
            nn.MaxPool2d(pool_size=(2, 2)),                       # Output (64, 5, 5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                     # Output (64 * 5 * 5 = 1600)
            nn.Dropout(0.5),
            nn.Linear(64 * 5 * 5, num_classes),
            nn.Softmax(dim=1) # Keep softmax as per original Keras model architecture
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_model(config: dict) -> nn.Module:
    """
    Instantiates and returns the PyTorch model.
    """
    model_kwargs = config.get("model_kwargs", {})
    return KerasLikeConvNet(num_classes=NUM_CLASSES, **model_kwargs)

# --------------------------------------------------------------------------------------------------
# 2. build_dataloader
# --------------------------------------------------------------------------------------------------

class SyntheticMNISTDataset(Dataset):
    """
    A synthetic dataset for MNIST-like data.
    Generates random images and one-hot encoded labels.
    """
    def __init__(self, num_samples: int, input_shape: tuple, num_classes: int):
        self.num_samples = num_samples
        # Convert Keras (H, W, C) to PyTorch (C, H, W)
        self.input_tensor_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.num_classes = num_classes

        self.data = torch.randn(num_samples, *self.input_tensor_shape, dtype=torch.float32)
        # Scale to [0, 1] as per original Keras preprocessing
        self.data = torch.abs(self.data) / self.data.max() if self.data.max() > 0 else self.data
        
        # Generate one-hot targets for consistency with original Keras setup
        self.targets = F.one_hot(
            torch.randint(0, num_classes, (num_samples,)),
            num_classes=num_classes
        ).to(torch.float32) # Ensure targets are float for potential loss calculations

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class RealMNISTDataset(Dataset):
    """
    Wrapper for torchvision.datasets.MNIST to handle data loading
    and preprocessing as per the original Keras script.
    """
    def __init__(self, data_path: str, train: bool, transform=None):
        try:
            # Check for common MNIST raw data files to verify data existence
            expected_data_dir = os.path.join(data_path, 'MNIST', 'raw')
            if not os.path.exists(expected_data_dir) or not any(
                f.startswith('train-images') or f.startswith('t10k-images')
                for f in os.listdir(expected_data_dir)
            ):
                raise FileNotFoundError(f"MNIST dataset raw files not found in {expected_data_dir}.")

            # Load the dataset using torchvision, but prevent download if not found
            self.mnist_dataset = datasets.MNIST(root=data_path, train=train, download=False, transform=transform)
        except RuntimeError as e:
            # torchvision.datasets.MNIST raises RuntimeError if download=False and data not found
            raise FileNotFoundError(f"MNIST dataset could not be loaded from {data_path}. Original error: {e}")

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img, target = self.mnist_dataset[idx]
        # Keras original had targets as one-hot. Convert here.
        target_one_hot = F.one_hot(torch.tensor(target, dtype=torch.int64), num_classes=NUM_CLASSES).to(torch.float32)
        return img, target_one_hot


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Returns a DataLoader for the requested split ("train" or "val").
    Supports synthetic data fallback.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    # Preprocessing: Keras original script only scaled to [0, 1]
    # ToTensor() converts PIL Image to (C, H, W) Tensor and scales pixels to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = None
    try:
        # Attempt to load the real dataset
        # The original Keras script used x_train for both training and validation split.
        # So we load the full "train" split of MNIST and then divide it.
        full_dataset = RealMNISTDataset(data_path=data_path, train=True, transform=transform)

        # Apply random_split as per Keras's validation_split=0.1
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        if split == "train":
            dataset = train_subset
        elif split == "val":
            dataset = val_subset
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

    except FileNotFoundError as e:
        if allow_synthetic_data:
            print(f"Warning: Real dataset not found at '{data_path}' ({e}). Using synthetic data for '{split}' split.")
            num_synthetic_samples = config.get("num_synthetic_samples", 1000)
            dataset = SyntheticMNISTDataset(
                num_samples=num_synthetic_samples,
                input_shape=INPUT_SHAPE,
                num_classes=NUM_CLASSES
            )
        else:
            raise FileNotFoundError(
                f"Real dataset not found at '{data_path}' and 'allow_synthetic_data' is False. "
                f"Please ensure the dataset is available or enable synthetic data. Original error: {e}"
            )

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


# --------------------------------------------------------------------------------------------------
# 3. train_step
# --------------------------------------------------------------------------------------------------
def train_step(model: nn.Module, batch, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Performs one forward pass and returns the loss.
    Does NOT call loss.backward() or optimizer.step().
    """
    # Move batch to model's device
    device = next(model.parameters()).device
    x, y_true = batch[0].to(device), batch[1].to(device)

    # Forward pass
    y_pred = model(x)

    # Calculate loss: Keras's categorical_crossentropy with softmax in the model
    # is equivalent to PyTorch's F.nll_loss if the model outputs log-probabilities
    # and targets are class indices.
    # Since our model outputs probabilities (due to nn.Softmax) and targets are one-hot,
    # we first take the log of probabilities (adding epsilon for numerical stability)
    # and convert one-hot targets to class indices (argmax).
    loss = F.nll_loss(torch.log(y_pred + 1e-10), y_true.argmax(dim=1))

    return loss
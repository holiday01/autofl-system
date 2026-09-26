import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import os

# Preserve original model architecture exactly, translated to PyTorch
class MNISTConvNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Input shape for Conv2d is (batch_size, channels, height, width)
        # For MNIST: (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)

        # Calculate input features for the dense layer dynamically
        # Let's assume an input of (1, 28, 28)
        # Conv1 output: (32, 28 - 3 + 1, 28 - 3 + 1) = (32, 26, 26)
        # Pool1 output: (32, 26 / 2, 26 / 2) = (32, 13, 13)
        # Conv2 output: (64, 13 - 3 + 1, 13 - 3 + 1) = (64, 11, 11)
        # Pool2 output: (64, 11 / 2 (floor), 11 / 2 (floor)) = (64, 5, 5)
        # Flattened size: 64 * 5 * 5 = 1600
        self.dense = nn.Linear(in_features=64 * 5 * 5, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x # Return logits, CrossEntropyLoss will apply softmax internally

def build_model(config: dict) -> torch.nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    num_classes = model_kwargs.get("num_classes", 10) # Default to 10 as in original MNIST
    return MNISTConvNet(num_classes=num_classes)

class SyntheticMNISTDataset(Dataset):
    def __init__(self, num_samples: int = 60000, img_shape: tuple = (1, 28, 28), num_classes: int = 10):
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.num_classes = num_classes
        # Synthetic images are random normal, matching expected float data type
        self.images = torch.randn(num_samples, *img_shape, dtype=torch.float32)
        # Synthetic labels are random integers within class range
        self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    # Original Keras script used validation_split=0.1 on its training data
    validation_split_ratio = config.get("validation_split_ratio", 0.1)

    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL Image to FloatTensor and scales to [0.0, 1.0]
        # Keras original also only scaled to [0,1], no further normalization specified.
    ])

    dataset = None
    try:
        # We load the full MNIST training set, then split it locally for client's train/val.
        full_train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

        # Apply random_split to create client-local training and validation sets
        train_size = int((1 - validation_split_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        # Adjust val_size to ensure sum matches original dataset length due to potential floating point issues
        if train_size + val_size != len(full_train_dataset):
            val_size = len(full_train_dataset) - train_size

        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        if split == "train":
            dataset = train_dataset
        elif split == "val":
            dataset = val_dataset
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

    except Exception as e:
        if allow_synthetic_data:
            print(f"WARNING: Could not load real data from {data_path} ({e}). Using synthetic data.")
            # Parameters for synthetic data, mimicking MNIST training set characteristics
            num_synthetic_samples = 60000 
            img_shape = (1, 28, 28)
            num_classes = config.get("model_kwargs", {}).get("num_classes", 10)
            
            synthetic_full_dataset = SyntheticMNISTDataset(
                num_samples=num_synthetic_samples, 
                img_shape=img_shape, 
                num_classes=num_classes
            )

            train_size = int((1 - validation_split_ratio) * len(synthetic_full_dataset))
            val_size = len(synthetic_full_dataset) - train_size
            
            # Adjust val_size for synthetic dataset
            if train_size + val_size != len(synthetic_full_dataset):
                val_size = len(synthetic_full_dataset) - train_size

            train_dataset, val_dataset = random_split(synthetic_full_dataset, [train_size, val_size])

            if split == "train":
                dataset = train_dataset
            elif split == "val":
                dataset = val_dataset
            else:
                raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")
        else:
            raise FileNotFoundError(
                f"Real dataset not found at '{data_path}' and 'allow_synthetic_data' is False. "
                f"Please ensure the dataset is available or enable synthetic data generation. "
                f"Original error: {type(e).__name__}: {e}"
            )

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))

def train_step(model: torch.nn.Module, batch: tuple, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    images, labels = batch
    
    # Get the device from the model's first parameter
    device = next(model.parameters()).device
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    outputs = model(images)

    # Calculate loss
    # The original Keras script used `categorical_crossentropy` with one-hot encoded labels.
    # PyTorch's `nn.CrossEntropyLoss` is the standard equivalent when model outputs logits
    # and labels are class indices (integers), which `torchvision.datasets.MNIST` provides.
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)

    return loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import os

# Preserve the original model architecture exactly, converted to PyTorch.
# The Keras model has Conv2D(..., activation="relu"), MaxPooling2D, Flatten, Dropout, Dense(..., activation="softmax").
# For PyTorch, the 'softmax' activation is typically omitted from the model's last layer
# if nn.CrossEntropyLoss is used, as CrossEntropyLoss internally applies log_softmax.
class MNIST_ConvNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Input shape: (N, 1, 28, 28) for PyTorch
        # Keras Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0) # Output (32, 26, 26)
        self.relu1 = nn.ReLU()
        # Keras MaxPooling2D(pool_size=(2, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output (32, 13, 13)

        # Keras Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0) # Output (64, 11, 11)
        self.relu2 = nn.ReLU()
        # Keras MaxPooling2D(pool_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output (64, 5, 5)

        # Keras Flatten() -> 64 * 5 * 5 = 1600 features
        self.flatten = nn.Flatten()

        # Keras Dropout(0.5)
        self.dropout = nn.Dropout(0.5)

        # Keras Dense(num_classes, activation="softmax")
        # In PyTorch, nn.CrossEntropyLoss expects raw logits, so no softmax here.
        self.dense = nn.Linear(64 * 5 * 5, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    - Use config.get("model_kwargs", {}) for constructor args.
    """
    model_kwargs = config.get("model_kwargs", {})
    num_classes = model_kwargs.get("num_classes", 10) # Default for MNIST
    model = MNIST_ConvNet(num_classes=num_classes)
    return model

class SyntheticMNISTDataset(Dataset):
    """
    A synthetic dataset for MNIST-like data.
    """
    def __init__(self, num_samples=1000, img_shape=(1, 28, 28), num_classes=10):
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.num_classes = num_classes
        # Generate random images (float values between 0 and 1)
        self.images = torch.rand(num_samples, *img_shape, dtype=torch.float32)
        # Generate random labels (long integers)
        self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    - Read batch_size from config.get("local", {}).get("batch_size", 16).
    - Read data_path from config.get("data_path", ".").
    - Use random_split to produce train/val subsets from a single dataset.
    - Include a synthetic data fallback.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    num_classes = config.get("model_kwargs", {}).get("num_classes", 10)

    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL Image to Tensor, scales to [0,1], and reshapes (H, W, C) to (C, H, W)
        # The original Keras script applies `astype("float32") / 255` and `np.expand_dims`.
        # ToTensor() handles scaling and channel dimension for PyTorch.
    ])

    try:
        # Ensure data_path exists for torchvision to download into
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        # Load the full MNIST training dataset
        full_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

        # Use random_split to create train and validation subsets
        # Original Keras script used `validation_split=0.1` during `model.fit`
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        if split == "train":
            dataset_to_load = train_dataset
        elif split == "val":
            dataset_to_load = val_dataset
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    except Exception as e:
        if allow_synthetic_data:
            print(f"WARNING: Could not load real MNIST data from '{data_path}'. Error: {e}. Using synthetic data.")
            if split == "train":
                dataset_to_load = SyntheticMNISTDataset(num_samples=50000, num_classes=num_classes)
            elif split == "val":
                dataset_to_load = SyntheticMNISTDataset(num_samples=10000, num_classes=num_classes)
            else:
                raise ValueError(f"Invalid split for synthetic data: {split}. Must be 'train' or 'val'.")
        else:
            raise FileNotFoundError(
                f"Real MNIST data not found at '{data_path}' and 'allow_synthetic_data' is False. "
                f"Error: {e}. Please ensure data is available or enable synthetic data."
            )

    shuffle = (split == "train") # Only shuffle training data
    dataloader = DataLoader(dataset_to_load, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return dataloader

def train_step(model: torch.nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    - Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    - Move tensors to the device of the model parameters.
    """
    # Determine the device from the model's parameters
    device = next(model.parameters()).device

    images, labels = batch
    images = images.to(device)
    # The Keras script used `keras.utils.to_categorical` for one-hot labels.
    # PyTorch's `nn.CrossEntropyLoss` expects raw class indices (long type) for labels.
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)

    # Loss function: Keras used 'categorical_crossentropy' with 'softmax' activation on the last layer.
    # In PyTorch, nn.CrossEntropyLoss combines LogSoftmax and NLLLoss,
    # and expects raw logits (output of the last linear layer) and integer class labels.
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)

    return loss
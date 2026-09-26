import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

# Attempt to import torchvision components, providing a fallback if not available
try:
    import torchvision
    from torchvision import transforms
    from torchvision.datasets import MNIST
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False
    transforms = None # Set to None if not available


class Backbone(torch.nn.Module):
    """
    The core neural network model for MNIST classification.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the image
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    The original script's `LitClassifier` wraps a `Backbone` model.
    For FL, we extract and federate the `Backbone` itself.
    """
    model_kwargs = config.get("model_kwargs", {})
    return Backbone(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Args:
        config: A dictionary containing configuration parameters.
                Expected keys:
                - "local": dict with "batch_size" (int)
                - "data_path": str (path to save/load dataset)
                - "allow_synthetic_data": bool (fallback to synthetic data if real not found)
        split: The requested data split, "train" or "val".

    Returns:
        A torch.utils.data.DataLoader for the specified split.

    Raises:
        ValueError: If an invalid split is requested.
        FileNotFoundError: If real data is not found and synthetic data is disallowed.
        ImportError: If torchvision is not available and synthetic data is disallowed.
    """
    if split not in ["train", "val"]:
        raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    transform = transforms.ToTensor() if _TORCHVISION_AVAILABLE else None

    # This will hold the full training dataset (equivalent to MNIST(train=True))
    data_source_dataset: Optional[Dataset] = None

    if _TORCHVISION_AVAILABLE:
        try:
            # Attempt to load the real MNIST training dataset
            data_source_dataset = MNIST(data_path, train=True, download=True, transform=transform)
        except Exception as e:
            if allow_synthetic_data:
                print(f"Warning: Could not load real MNIST dataset ({e}). Generating synthetic data.")
                # data_source_dataset remains None, proceeding to synthetic data generation
            else:
                raise FileNotFoundError(
                    f"Real dataset not found at '{data_path}' and 'allow_synthetic_data' is False. "
                    "Please ensure the dataset is available or set 'allow_synthetic_data' to True in config."
                )
    else: # _TORCHVISION_AVAILABLE is False
        if allow_synthetic_data:
            print("Warning: torchvision is not available. Generating synthetic data.")
            # data_source_dataset remains None, proceeding to synthetic data generation
        else:
            raise ImportError(
                "torchvision is not available and 'allow_synthetic_data' is False. "
                "Please install torchvision or set 'allow_synthetic_data' to True in config."
            )

    # If real data couldn't be loaded or torchvision is not available, and synthetic data is allowed
    if data_source_dataset is None:
        # Generate synthetic data for the 'full_train_dataset' equivalent
        num_samples = 60000  # Standard MNIST training set size
        synthetic_images = torch.randn(num_samples, 1, 28, 28)  # MNIST images are 1 channel 28x28
        synthetic_labels = torch.randint(0, 10, (num_samples,))  # 10 classes
        data_source_dataset = TensorDataset(synthetic_images, synthetic_labels)

    # Perform random_split to get train/val subsets from the data_source_dataset
    # The original script split 60000 samples into 55000 train, 5000 val
    total_len = len(data_source_dataset)
    # Calculate partition sizes based on the original proportions
    train_len = int(total_len * (55000 / (55000 + 5000)))
    val_len = total_len - train_len

    train_subset, val_subset = random_split(
        data_source_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    if split == "train":
        dataset_for_split = train_subset
        shuffle_data = True  # Typically shuffle training data
    else:  # split == "val"
        dataset_for_split = val_subset
        shuffle_data = False  # Typically do not shuffle validation data

    return DataLoader(dataset_for_split, batch_size=batch_size, shuffle=shuffle_data)


def train_step(model: torch.nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step().

    Args:
        model: The PyTorch model to train.
        batch: A tuple (x, y) containing input features and target labels.
        optimizer: The optimizer (not used directly here, but required by API).
        config: A dictionary containing configuration parameters (not used directly here).

    Returns:
        A scalar loss tensor with gradient information attached.
    """
    x, y = batch

    # Move tensors to the model's device
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)

    # Forward pass
    y_hat = model(x)
    # Calculate loss (F.cross_entropy from LitClassifier's training_step)
    loss = F.cross_entropy(y_hat, y)
    return loss
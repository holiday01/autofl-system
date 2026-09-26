import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import StepLR


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


def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the Net model.

    Args:
        config: FL configuration dict. Supports key 'model_kwargs' (dict)
                forwarded to the Net constructor (currently unused by Net,
                kept for forward-compatibility).

    Returns:
        An initialised Net instance (CPU by default; the FL runtime moves it).
    """
    model_kwargs = config.get("model_kwargs", {})
    model = Net(**model_kwargs)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    The full MNIST *training* set is loaded once and divided 80/20 into
    train and val subsets via random_split (seeded for reproducibility).

    Args:
        config: FL configuration dict with optional keys:
            - local.batch_size   (int, default 16)
            - data_path          (str, default ".")
            - allow_synthetic_data (bool, default False)
        split: "train" or "val".

    Returns:
        A DataLoader over the requested split.

    Raises:
        FileNotFoundError: If the MNIST dataset cannot be loaded and
                           config['allow_synthetic_data'] is False.
        ValueError: If split is not "train" or "val".
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_dataset = None
    load_error = None

    try:
        # Attempt to load real MNIST (download if absent).
        full_dataset = datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transform,
        )
    except Exception as exc:
        load_error = exc

    if full_dataset is None:
        # Real data unavailable — synthetic fallback only if explicitly allowed.
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset could not be loaded from '{data_path}' "
                f"(original error: {load_error}). "
                "To use synthetic data instead, set config['allow_synthetic_data'] = True."
            )
        # Synthetic: 1 000 grayscale 28×28 images, 10 classes.
        n_synthetic = 1000
        synthetic_images = torch.randn(n_synthetic, 1, 28, 28)
        synthetic_labels = torch.randint(0, 10, (n_synthetic,))
        full_dataset = TensorDataset(synthetic_images, synthetic_labels)

    # Deterministic 80 / 20 split — same partition for every client call.
    total = len(full_dataset)
    val_size = max(1, int(round(0.2 * total)))
    train_size = total - val_size
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_subset if split == "train" else val_subset
    shuffle = split == "train"

    return DataLoader(chosen, batch_size=batch_size, shuffle=shuffle)


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run a single forward pass and return the loss tensor.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.

    Args:
        model:     The Net instance (already on the target device).
        batch:     A (data, target) tuple as yielded by build_dataloader.
        optimizer: The optimizer (not stepped here).
        config:    FL configuration dict (reserved for future use).

    Returns:
        The scalar loss tensor with an attached grad_fn.
    """
    device = next(model.parameters()).device

    data, target = batch
    data = data.to(device)
    target = target.to(device)

    model.train()
    output = model(data)
    loss = F.nll_loss(output, target)
    return loss
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, TensorDataset


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
    model_kwargs = config.get("model_kwargs", {})
    model = Net(**model_kwargs)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = None
    try:
        full_dataset = datasets.MNIST(
            data_path, train=True, download=False, transform=transform
        )
        # Trigger actual file access to confirm data is readable
        _ = full_dataset[0]
    except Exception as exc:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset not found or unreadable at '{data_path}'. "
                "Pre-download the dataset or set config['allow_synthetic_data'] = True "
                "to fall back to synthetic data for testing purposes."
            ) from exc
        # Synthetic fallback: 1 000 random (image, label) pairs matching MNIST dims
        n_samples = 1000
        synthetic_images = torch.randn(n_samples, 1, 28, 28)
        synthetic_labels = torch.randint(0, 10, (n_samples,))
        full_dataset = TensorDataset(synthetic_images, synthetic_labels)

    n_total = len(full_dataset)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_subset if split == "train" else val_subset
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    device = next(model.parameters()).device
    data, target = batch
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    loss = F.nll_loss(output, target)
    return loss
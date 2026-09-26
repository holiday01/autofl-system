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
    return Net(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        full_dataset = datasets.MNIST(
            data_path, train=True, download=True, transform=transform
        )
        val_size = max(1, int(0.1 * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        train_subset, val_subset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    except Exception:
        # Synthetic fallback: 1-channel 28x28 images, 10 classes
        n_total = 1000
        data = torch.randn(n_total, 1, 28, 28)
        targets = torch.randint(0, 10, (n_total,))
        full_dataset = TensorDataset(data, targets)
        val_size = max(1, int(0.1 * n_total))
        train_size = n_total - val_size
        train_subset, val_subset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    chosen = train_subset if split == "train" else val_subset
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    data, target = batch
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    loss = F.nll_loss(output, target)
    return loss
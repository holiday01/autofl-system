import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split


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


def build_model(config: dict) -> nn.Module:
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
            data_path,
            train=True,
            download=False,
            transform=transform,
        )
    except Exception as exc:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset not found at '{data_path}'. "
                "Provide the data at that path or set config['allow_synthetic_data'] = True "
                "to allow synthetic data (for smoke-testing only)."
            ) from exc
        n_samples = 1000
        synthetic_images = torch.randn(n_samples, 1, 28, 28)
        synthetic_labels = torch.randint(0, 10, (n_samples,))
        full_dataset = TensorDataset(synthetic_images, synthetic_labels)

    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    if split == "train":
        chosen, shuffle = train_dataset, True
    elif split == "val":
        chosen, shuffle = val_dataset, False
    else:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    return DataLoader(chosen, batch_size=batch_size, shuffle=shuffle)


def train_step(
    model: nn.Module,
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
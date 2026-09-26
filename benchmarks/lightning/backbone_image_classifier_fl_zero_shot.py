import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


def build_model(config: dict) -> torch.nn.Module:
    hidden_dim = config.get("hidden_dim", 128)
    return Backbone(hidden_dim=hidden_dim)


def build_dataloader(config: dict, split: str) -> DataLoader:
    data_dir = config.get("data_dir", "./data")
    batch_size = config.get("batch_size", 32)
    seed = config.get("seed", 42)

    transform = transforms.ToTensor()

    if split == "train":
        dataset = MNIST(data_dir, train=True, download=True, transform=transform)
        train_set, _ = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(seed)
        )
        return DataLoader(train_set, batch_size=batch_size, shuffle=True)
    elif split == "val":
        dataset = MNIST(data_dir, train=True, download=True, transform=transform)
        _, val_set = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(seed)
        )
        return DataLoader(val_set, batch_size=batch_size, shuffle=False)
    elif split == "test":
        dataset = MNIST(data_dir, train=False, download=True, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")


def train_step(
    model: torch.nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> dict:
    model.train()
    x, y = batch
    optimizer.zero_grad()
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    loss.backward()
    optimizer.step()
    return {"loss": loss.item()}
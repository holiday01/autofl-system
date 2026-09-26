from __future__ import annotations

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def _loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def build_model(config: dict) -> nn.Module:
    return VAE()


def build_dataloader(config: dict, split: str) -> torch.utils.data.DataLoader:
    assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"

    batch_size = config.get("batch_size", 128)
    data_dir = config.get("data_dir", "../data")
    use_accel = config.get("use_accel", torch.accelerator.is_available() if hasattr(torch, "accelerator") else False)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_accel else {}
    is_train = split == "train"

    dataset = datasets.MNIST(
        data_dir,
        train=is_train,
        download=True,
        transform=transforms.ToTensor(),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        **kwargs,
    )


def train_step(
    model: nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> dict:
    data, _ = batch
    device = next(model.parameters()).device
    data = data.to(device)

    model.train()
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data)
    loss = _loss_function(recon_batch, data, mu, logvar)
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "num_samples": data.size(0)}
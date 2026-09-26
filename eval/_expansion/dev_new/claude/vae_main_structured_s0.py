from __future__ import print_function
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

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
    # Reconstruction + KL divergence losses summed over all elements and batch
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def build_model(config: dict) -> nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    model = VAE(**model_kwargs)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    if split not in ("train", "val"):
        raise ValueError(f"Unknown split '{split}'. Expected 'train' or 'val'.")

    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    try:
        full_dataset = datasets.MNIST(
            data_path,
            train=True,
            download=False,
            transform=transforms.ToTensor(),
        )
    except (RuntimeError, FileNotFoundError):
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset not found at '{data_path}'. "
                "Provide the correct 'data_path' in config, or set "
                "config['allow_synthetic_data'] = True to use a synthetic "
                "fallback for smoke-testing only."
            )
        # Synthetic fallback: pixel values in [0, 1] to match ToTensor() output
        # and remain compatible with the VAE's BCE loss and sigmoid decoder.
        _n = 1000
        images = torch.rand(_n, 1, 28, 28)
        labels = torch.randint(0, 10, (_n,))
        full_dataset = TensorDataset(images, labels)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_dataset if split == "train" else val_dataset
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    device = next(model.parameters()).device

    data, _ = batch
    data = data.to(device)

    # Single forward pass — backward() and optimizer.step() are handled by the FL runtime
    recon_batch, mu, logvar = model(data)
    loss = _loss_function(recon_batch, data, mu, logvar)

    return loss  # grad is attached; caller drives backward + step

---

Key decisions made during the conversion:

| Concern | Decision |
|---|---|
| **Model** | `VAE` class copied verbatim; `build_model` passes `**config.get("model_kwargs", {})` (empty by default, harmless since `__init__` takes no args) |
| **Loss** | `_loss_function` (prefixed with `_`) preserves the exact BCE + KLD formula; private so the FL runtime doesn't accidentally call it directly |
| **DataLoader splits** | One `datasets.MNIST(train=True)` load → 90/10 `random_split` with a fixed seed for reproducibility across train/val |
| **`download=False`** | Real FL clients should receive pre-staged data; auto-download is suppressed so missing data fails loudly |
| **Synthetic fallback** | Uses `torch.rand` (not `randn`) to stay in `[0, 1]` — required by the VAE's sigmoid decoder and BCE loss; strictly gated on `allow_synthetic_data` |
| **`train_step`** | Calls only the forward pass and returns the live loss tensor; no `loss.backward()` or `optimizer.step()` |
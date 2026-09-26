"""
FL client for mnist_lite.py (Lightning GAN) — zero-shot style.
Demonstrates FL contract violation: backward/step called inside train_step,
and no synthetic data fallback in build_dataloader.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(math.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(math.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))


class GANModel(nn.Module):
    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)
        self.latent_dim = latent_dim

    def forward(self, z):
        return self.generator(z)


def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return GANModel(
        latent_dim=kwargs.get("latent_dim", 100),
        img_shape=tuple(kwargs.get("img_shape", [1, 28, 28])),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    # Zero-shot style: no synthetic fallback — will fail if real data unavailable
    from torchvision import datasets, transforms
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)
    dataset = config.get("dataset", {"data_path": data_path})
    full_ds = datasets.MNIST(
        dataset["data_path"], train=(split == "train"), download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(full_ds, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """Zero-shot style: calls backward/step inside — FL contract violation."""
    device = next(model.parameters()).device
    imgs, _ = batch
    imgs = imgs.to(device)
    batch_size = imgs.size(0)

    valid = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    # Train discriminator
    z = torch.randn(batch_size, model.latent_dim, device=device)
    fake_imgs = model.generator(z).detach()
    real_loss = F.binary_cross_entropy_with_logits(model.discriminator(imgs), valid)
    fake_loss = F.binary_cross_entropy_with_logits(model.discriminator(fake_imgs), fake_labels)
    d_loss = (real_loss + fake_loss) / 2

    if optimizer is not None:
        optimizer.zero_grad()
        d_loss.backward()
        optimizer.step()

    # Train generator
    z = torch.randn(batch_size, model.latent_dim, device=device)
    g_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(model.generator(z)), valid
    )
    if optimizer is not None:
        optimizer.zero_grad()
        g_loss.backward()
        optimizer.step()

    return (d_loss + g_loss).detach()

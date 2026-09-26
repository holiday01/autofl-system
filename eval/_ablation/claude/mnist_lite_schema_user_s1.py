import math
import os
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import torchvision
    import torchvision.transforms as transforms
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


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
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(math.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)


class GANModel(nn.Module):
    def __init__(self, img_shape: tuple = (1, 28, 28), latent_dim: int = 100):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

    def forward(self, z):
        return self.generator(z)


def build_model(config: dict) -> torch.nn.Module:
    kwargs = config.get("model_kwargs", {})
    img_shape = kwargs.get("img_shape", (1, 28, 28))
    latent_dim = kwargs.get("latent_dim", 100)
    return GANModel(img_shape=img_shape, latent_dim=latent_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    dataset = None

    if _TORCHVISION_AVAILABLE:
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=True,
                download=False,
                transform=transform,
            )
        except Exception:
            dataset = None

    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset not found at '{data_path}'. "
                "Ensure the data exists at that path or set "
                "config['allow_synthetic_data'] = True to use synthetic data."
            )
        n_samples = 1000
        imgs = torch.randn(n_samples, 1, 28, 28)
        labels = torch.randint(0, 10, (n_samples,))
        dataset = TensorDataset(imgs, labels)

    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_subset, val_subset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_subset if split == "train" else val_subset
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: torch.nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    imgs, _ = batch
    device = next(model.parameters()).device
    imgs = imgs.to(device)

    batch_size = imgs.size(0)
    latent_dim = model.latent_dim

    z = torch.randn(batch_size, latent_dim, device=device)
    valid = torch.ones(batch_size, 1, device=device)

    # Generator loss: generator tries to fool the discriminator
    g_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(model.generator(z)),
        valid,
    )

    # Discriminator loss: distinguish real images from generated ones
    real_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(imgs),
        valid,
    )
    fake = torch.zeros(batch_size, 1, device=device)
    # detach generator output so discriminator gradients don't flow into generator here
    fake_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(model.generator(z).detach()),
        fake,
    )
    d_loss = (real_loss + fake_loss) / 2

    return g_loss + d_loss
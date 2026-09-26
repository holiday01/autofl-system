import math
import os

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
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)


def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return GANModel(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    if _TORCHVISION_AVAILABLE:
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            full_dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=True,
                download=False,
                transform=transform,
            )
            n_val = max(1, int(0.1 * len(full_dataset)))
            n_train = len(full_dataset) - n_val
            train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
            ds = train_ds if split == "train" else val_ds
            return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
        except Exception:
            pass

    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"MNIST dataset not found at '{data_path}'. "
            "Set config['allow_synthetic_data'] = True to use synthetic data instead."
        )

    img_shape = config.get("model_kwargs", {}).get("img_shape", (1, 28, 28))
    n_samples = 1000
    imgs = torch.randn(n_samples, *img_shape)
    labels = torch.randint(0, 10, (n_samples,))
    full_dataset = TensorDataset(imgs, labels)
    n_val = max(1, int(0.1 * n_samples))
    n_train = n_samples - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
    ds = train_ds if split == "train" else val_ds
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    imgs, _ = batch
    imgs = imgs.to(device)

    z = torch.randn(imgs.size(0), model.latent_dim, device=device)
    valid = torch.ones(imgs.size(0), 1, device=device)
    fake = torch.zeros(imgs.size(0), 1, device=device)

    g_loss = model.adversarial_loss(model.discriminator(model.generator(z)), valid)

    real_loss = model.adversarial_loss(model.discriminator(imgs), valid)
    fake_imgs = model.generator(z).detach()
    fake_loss = model.adversarial_loss(model.discriminator(fake_imgs), fake)
    d_loss = (real_loss + fake_loss) / 2

    return g_loss + d_loss
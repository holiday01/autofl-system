import math
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from lightning.pytorch import cli_lightning_logo
from lightning.pytorch.core import LightningModule
from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision import datasets, transforms


class Generator(nn.Module):
    """
    >>> Generator(img_shape=(1, 8, 8))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Generator(
      (model): Sequential(...)
    )
    """

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
    """
    >>> Discriminator(img_shape=(1, 28, 28))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Discriminator(
      (model): Sequential(...)
    )
    """

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


class GAN(LightningModule):
    """
    >>> GAN(img_shape=(1, 8, 8))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GAN(
      (generator): Generator(
        (model): Sequential(...)
      )
      (discriminator): Discriminator(
        (model): Sequential(...)
      )
    )
    """

    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        latent_dim: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch):
        imgs, _ = batch

        opt_g, opt_d = self.optimizers()

        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        self.toggle_optimizer(opt_g)
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        self.toggle_optimizer(opt_d)
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.log_dict({"d_loss": d_loss, "g_loss": g_loss})

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return opt_g, opt_d

    def on_train_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        for logger in self.loggers:
            logger.experiment.add_image("generated_images", grid, self.current_epoch)


class GANModel(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), latent_dim=100):
        super().__init__()
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)
        self.latent_dim = latent_dim

    def forward(self, z):
        return self.generator(z)


def build_model(config):
    img_shape = tuple(config.get("img_shape", (1, 28, 28)))
    latent_dim = config.get("latent_dim", 100)
    return GANModel(img_shape=img_shape, latent_dim=latent_dim)


def build_dataloader(config, split="train"):
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    full_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    n_val = max(1, int(len(full_dataset) * 0.2))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    chosen = train_set if split == "train" else val_set
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config):
    imgs, _ = batch
    opt_g, opt_d = optimizer
    latent_dim = config.get("latent_dim", model.latent_dim)

    z = torch.randn(imgs.shape[0], latent_dim, device=imgs.device, dtype=imgs.dtype)

    valid_g = torch.ones(imgs.size(0), 1, device=imgs.device, dtype=imgs.dtype)
    g_loss = F.binary_cross_entropy_with_logits(model.discriminator(model.generator(z)), valid_g)
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()

    z = torch.randn(imgs.shape[0], latent_dim, device=imgs.device, dtype=imgs.dtype)
    valid_d = torch.ones(imgs.size(0), 1, device=imgs.device, dtype=imgs.dtype)
    real_loss = F.binary_cross_entropy_with_logits(model.discriminator(imgs), valid_d)

    fake_labels = torch.zeros(imgs.size(0), 1, device=imgs.device, dtype=imgs.dtype)
    fake_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(model.generator(z).detach()), fake_labels
    )

    d_loss = (real_loss + fake_loss) / 2
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()

    return {"d_loss": d_loss.item(), "g_loss": g_loss.item()}
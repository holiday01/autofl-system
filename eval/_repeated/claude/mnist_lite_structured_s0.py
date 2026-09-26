import math
import os
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

try:
    from lightning.pytorch import cli_lightning_logo
    from lightning.pytorch.core import LightningModule
    from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
    from lightning.pytorch.trainer import Trainer
    from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE as _LPL_TV_FLAG
except ImportError:
    _LPL_TV_FLAG = None

try:
    import torchvision
    import torchvision.transforms as transforms
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

if _LPL_TV_FLAG is not None:
    _TORCHVISION_AVAILABLE = bool(_LPL_TV_FLAG) and _TORCHVISION_AVAILABLE


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


class GANModule(nn.Module):
    """Combined GAN module wrapping Generator and Discriminator for FL training.

    Replaces the LightningModule-based GAN with a plain nn.Module that exposes
    both sub-networks under a single parameter namespace so the FL runtime's
    single optimizer can update them in one step.
    """

    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
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


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return a GANModule (Generator + Discriminator).

    Recognised config.model_kwargs keys
    ------------------------------------
    latent_dim : int   (default 100)
    img_shape  : list  (default [1, 28, 28])
    """
    kwargs = config.get("model_kwargs", {})
    latent_dim = kwargs.get("latent_dim", 100)
    img_shape = tuple(kwargs.get("img_shape", (1, 28, 28)))
    return GANModule(latent_dim=latent_dim, img_shape=img_shape)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    Tries to load MNIST from *data_path* (download=True so a fresh client
    can self-provision).  On any failure the behaviour depends on the flag
    config['allow_synthetic_data']:
      - False (default) → raises FileNotFoundError immediately.
      - True            → falls back to a TensorDataset of random noise.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    real_data_error: Exception = ImportError(
        "torchvision is not installed; cannot load MNIST."
    )

    if _TORCHVISION_AVAILABLE:
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            full_dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=True,
                download=True,
                transform=transform,
            )
            n_val = max(1, int(0.1 * len(full_dataset)))
            n_train = len(full_dataset) - n_val
            train_set, val_set = random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            chosen = train_set if split == "train" else val_set
            return DataLoader(
                chosen,
                batch_size=batch_size,
                shuffle=(split == "train"),
                drop_last=True,
            )
        except Exception as exc:
            real_data_error = exc

    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"MNIST dataset could not be loaded from '{data_path}' "
            f"(reason: {real_data_error}). "
            "Provide a valid data_path that contains (or can download) MNIST, "
            "or set config['allow_synthetic_data'] = True to allow synthetic "
            "stand-in data for smoke-testing."
        )

    # Synthetic fallback — reached only when allow_synthetic_data is True.
    n_samples = 1000 if split == "train" else 200
    imgs = torch.randn(n_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (n_samples,))
    synthetic_dataset = TensorDataset(imgs, labels)
    return DataLoader(
        synthetic_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=True,
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """Single forward pass returning a combined GAN loss (grad attached).

    GAN-specific note
    -----------------
    The original script uses two separate optimizers with manual_backward.
    The FL runtime supplies a single optimizer that covers all parameters of
    the GANModule.  We therefore compute a combined loss:

        loss = g_loss + d_loss

    Gradient routing is preserved by detaching fake_imgs before feeding them
    to the discriminator for d_loss, exactly as in the original:

    * g_loss  → gradients flow through discriminator → generator  (generator update)
    * d_loss  → gradients flow through discriminator only          (discriminator update)
                (fake_imgs is detached, so generator is not touched twice)

    The FL runtime calls loss.backward() and optimizer.step() after this returns;
    this function must NOT do so.
    """
    device = next(model.parameters()).device
    imgs, _ = batch
    imgs = imgs.to(device)

    latent_dim = model.latent_dim

    # Sample noise — same pattern as the original training_step.
    z = torch.randn(imgs.size(0), latent_dim, device=device)

    # ---- Generator loss ------------------------------------------------
    # Generator wants the discriminator to output 1 ("valid") for fake images.
    fake_imgs = model.generator(z)
    valid = torch.ones(imgs.size(0), 1, device=device)
    g_loss = model.adversarial_loss(model.discriminator(fake_imgs), valid)

    # ---- Discriminator loss --------------------------------------------
    # Real images should be classified as valid (1).
    real_loss = model.adversarial_loss(model.discriminator(imgs), valid)
    # Fake images (detached) should be classified as fake (0).
    fake_labels = torch.zeros(imgs.size(0), 1, device=device)
    fake_loss = model.adversarial_loss(
        model.discriminator(fake_imgs.detach()), fake_labels
    )
    d_loss = (real_loss + fake_loss) / 2

    # Combined loss for single-optimizer FL step.
    loss = g_loss + d_loss
    return loss
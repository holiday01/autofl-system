"""
Auto-generated FL client module.
Original script: generative_adversarial_net.py — PyTorch Lightning GAN
                 LightningModule extracted into plain nn.Module for federated training.

Exposes:
  build_model(config)                       -> nn.Module
  build_dataloader(config, split)           -> DataLoader
  train_step(model, batch, opt, config)     -> loss tensor (with grad_fn)

CONTRACT:
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach() on it).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.

NOTE (GAN):
  Standard GAN training alternates generator and discriminator updates with two
  separate optimizers.  The FL runtime expects a single loss scalar and a single
  optimizer, so train_step returns  loss = g_loss + d_loss.
  g_loss differentiates through the generator; d_loss uses a detached copy of
  the generated images so the discriminator receives a clean gradient signal
  without also pushing the generator toward the wrong objective.
"""

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


# ── Preserved original architecture ──────────────────────────────────────────

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


class GANModel(nn.Module):
    """Thin wrapper combining Generator + Discriminator for single-model FL training.

    The LightningModule's configure_optimizers / manual_backward / toggle_optimizer
    machinery is intentionally omitted — the FL runtime owns all of that.
    """

    def __init__(self, img_shape: tuple = (1, 28, 28), latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

    def forward(self, z):
        """Delegate forward to the generator (image sampling / inference)."""
        return self.generator(z)


# ── FL Interface ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return GANModel(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 16))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)
    data_path   = config.get("data_path", ".")
    val_ratio   = config.get("val_ratio", 0.1)

    full_dataset = None

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
        except Exception:
            full_dataset = None

    if full_dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset not found at '{data_path}' and "
                f"torchvision is {'installed but the data files are missing' if _TORCHVISION_AVAILABLE else 'not installed'}. "
                "Either (a) set config['data_path'] to a directory that already "
                "contains the MNIST raw files, (b) install torchvision and "
                "pre-download MNIST to that path, or (c) set "
                "config['allow_synthetic_data'] = True to use random tensors "
                "for smoke-testing only — never for real training."
            )
        model_kwargs = config.get("model_kwargs", {})
        img_shape = model_kwargs.get("img_shape", (1, 28, 28))
        n = config.get("synthetic_n", 200)
        X = torch.randn(n, *img_shape)
        y = torch.randint(0, 10, (n,))
        full_dataset = TensorDataset(X, y)

    n_val   = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
    )
    ds = train_ds if split == "train" else val_ds
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass covering both generator and discriminator.
    Returns  loss = g_loss + d_loss  with grad_fn attached.

    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.

    GAN note:
      g_loss  — adversarial loss for the generator; gradients flow back through
                the generator weights via the discriminator's scoring of fake imgs.
      d_loss  — adversarial loss for the discriminator computed with a *detached*
                copy of the generated images, so the discriminator receives a clean
                gradient signal without also penalising the generator for the
                discriminator task.  This mirrors the toggle_optimizer / manual_backward
                separation in the original LightningModule.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        imgs = batch[0].to(device)
    elif isinstance(batch, dict):
        key = next(k for k in ("image", "x", "input") if k in batch)
        imgs = batch[key].to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    batch_size = imgs.size(0)
    latent_dim = model.latent_dim

    # Sample noise on the correct device
    z = torch.randn(batch_size, latent_dim, device=device)

    # ── Generator loss ────────────────────────────────────────────────────────
    # Generator wants discriminator to classify its fakes as real (label = 1)
    fake_imgs = model.generator(z)
    valid = torch.ones(batch_size, 1, device=device)
    g_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs), valid
    )

    # ── Discriminator loss ────────────────────────────────────────────────────
    # Real images should be classified as real (label = 1)
    real_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(imgs), valid
    )
    # Fake images (detached) should be classified as fake (label = 0)
    fake_labels = torch.zeros(batch_size, 1, device=device)
    fake_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs.detach()), fake_labels
    )
    d_loss = (real_loss + fake_loss) / 2

    # Combined scalar — grad_fn attached — returned to the FL runtime
    loss = g_loss + d_loss
    return loss
"""
Auto-generated FL client module.
Original script: generative_adversarial_net.py

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.

NOTE (GAN specifics):
  The returned loss is g_loss + d_loss computed in a single forward pass.
  Generator loss flows through discriminator(generator(z)).
  Discriminator loss uses detached fake images so discriminator gradients do
  not flow back through the generator a second time — only g_loss updates the
  generator. A single shared optimizer over all parameters is assumed.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from torchvision import transforms
    from torchvision.datasets import MNIST
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
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)


class GANWrapper(nn.Module):
    """Wraps Generator and Discriminator into a single nn.Module for the FL interface."""

    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

    def forward(self, z):
        return self.generator(z)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    latent_dim = kwargs.get("latent_dim", 100)
    img_shape = tuple(kwargs.get("img_shape", [1, 28, 28]))
    return GANWrapper(latent_dim=latent_dim, img_shape=img_shape)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory = local.get("pin_memory", True)
    data_path = config.get("data_path", "./data")

    if _TORCHVISION_AVAILABLE:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        is_train = split != "test"
        full_dataset = MNIST(root=data_path, train=is_train, download=True, transform=transform)
        if is_train:
            val_ratio = config.get("val_ratio", 0.1)
            n_val = max(1, int(len(full_dataset) * val_ratio))
            n_train = len(full_dataset) - n_val
            train_ds, val_ds = random_split(
                full_dataset, [n_train, n_val],
                generator=torch.Generator().manual_seed(config.get("seed", 42)),
            )
            ds = train_ds if split == "train" else val_ds
        else:
            ds = full_dataset
    else:
        # synthetic fallback
        img_shape = tuple(config.get("model_kwargs", {}).get("img_shape", [1, 28, 28]))
        n = 1000 if split == "train" else 200
        X = torch.randn(n, *img_shape)
        y = torch.zeros(n, dtype=torch.long)
        ds = TensorDataset(X, y)

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
    ONE forward pass for GAN training.  Returns g_loss + d_loss WITH grad_fn.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.

    g_loss: adversarial loss pushing the generator to fool the discriminator.
    d_loss: average of real and fake classification losses for the discriminator.
    Fake images used in d_loss are detached so discriminator gradients do not
    propagate through the generator a second time.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        imgs = batch[0]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        imgs = batch.get("image", batch.get("x", batch.get("input")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    generator = model.generator
    discriminator = model.discriminator
    latent_dim = model.latent_dim

    z = torch.randn(imgs.size(0), latent_dim, device=device, dtype=imgs.dtype)

    valid = torch.ones(imgs.size(0), 1, device=device, dtype=imgs.dtype)
    fake = torch.zeros(imgs.size(0), 1, device=device, dtype=imgs.dtype)

    # Generator loss: generated images should be classified as real
    g_loss = F.binary_cross_entropy_with_logits(discriminator(generator(z)), valid)

    # Discriminator loss: real images as real, detached fakes as fake
    real_loss = F.binary_cross_entropy_with_logits(discriminator(imgs), valid)
    # detach here prevents a second generator gradient path through d_loss;
    # the generator is already updated via g_loss above
    fake_loss = F.binary_cross_entropy_with_logits(discriminator(generator(z).detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    return g_loss + d_loss
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

NOTE (GAN-specific):
  GANs canonically require two alternating optimizers (one for G, one for D).
  This module collapses both losses into a single combined loss
  (g_loss + d_loss) compatible with the single-optimizer FL contract.
  The discriminator sub-loss uses detached generator output so D gradients
  do not flow back through G within the same backward pass — this is the
  standard GAN detach trick, not a violation of the contract.
  For production FL-GAN training with alternating optimizers, build a
  custom runtime that calls train_step_g / train_step_d separately.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


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


class GANModel(nn.Module):
    """Wraps Generator + Discriminator as a single nn.Module for the FL interface."""

    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

    def forward(self, z):
        return self.generator(z)


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    latent_dim = kwargs.get("latent_dim", 100)
    img_shape = tuple(kwargs.get("img_shape", (1, 28, 28)))
    return GANModel(latent_dim=latent_dim, img_shape=img_shape)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    data_path = config.get("data_path", ".")
    val_ratio = config.get("val_ratio", 0.1)

    try:
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        full_dataset = torchvision.datasets.MNIST(
            root=data_path, train=True, download=True, transform=transform
        )
    except Exception:
        n = 1000
        X = torch.rand(n, 1, 28, 28) * 2 - 1
        y = torch.zeros(n, dtype=torch.long)
        full_dataset = TensorDataset(X, y)

    n_val = max(1, int(len(full_dataset) * val_ratio))
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
        drop_last=True,
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE combined GAN forward pass.  Returns g_loss + d_loss WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT .item() the returned loss.

    The discriminator sub-loss uses fake_imgs.detach() so that D gradients
    do not flow back through G in the same backward pass — this is the
    standard GAN detach trick, not a violation of the no-detach contract
    (the *returned* loss is never detached).
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        real_imgs = batch[0]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        real_imgs = batch.get("image", batch.get("x", batch.get("input")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    batch_size = real_imgs.size(0)
    z = torch.randn(batch_size, model.latent_dim, device=device, dtype=real_imgs.dtype)
    fake_imgs = model.generator(z)

    valid = torch.ones(batch_size, 1, device=device, dtype=real_imgs.dtype)
    fake_labels = torch.zeros(batch_size, 1, device=device, dtype=real_imgs.dtype)

    g_loss = F.binary_cross_entropy_with_logits(model.discriminator(fake_imgs), valid)

    real_loss = F.binary_cross_entropy_with_logits(model.discriminator(real_imgs), valid)
    fake_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs.detach()), fake_labels
    )
    d_loss = (real_loss + fake_loss) / 2

    return g_loss + d_loss
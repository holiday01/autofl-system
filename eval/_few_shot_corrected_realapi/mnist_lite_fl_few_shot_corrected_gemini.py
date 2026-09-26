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
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Assuming torchvision is available, as it's used in the original script
import torchvision
import torchvision.transforms as transforms


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


# Wrapper for Generator and Discriminator to act as a single nn.Module
class GANModel(nn.Module):
    def __init__(self, img_shape: tuple = (1, 28, 28), latent_dim: int = 100):
        super().__init__()
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)
        self.img_shape = img_shape
        self.latent_dim = latent_dim

    def forward(self, z):
        # The forward method of the GANModel will typically be used for the generator
        # when generating samples, similar to how the original GAN.forward() worked.
        return self.generator(z)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    # Default values from original script if not in config
    img_shape = kwargs.get("img_shape", (1, 28, 28))
    latent_dim = kwargs.get("latent_dim", 100)
    return GANModel(img_shape=img_shape, latent_dim=latent_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32)) # Original script uses 32 for MNIST
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)
    data_path = config.get("data_path", "./data") # Default data path

    # Transforms for MNIST, normalizing to [-1, 1] as required by GANs
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1]
    ])

    if split == "train" or split == "val":
        full_train_dataset = torchvision.datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transform,
        )
        val_ratio = config.get("val_ratio", 0.1)
        n_total = len(full_train_dataset)
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            full_train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.get("seed", 42)),
        )
        ds = train_ds if split == "train" else val_ds
    elif split == "test":
        ds = torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"), # Only shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def adversarial_loss(y_hat, y):
    return F.binary_cross_entropy_with_logits(y_hat, y)


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer, # This will be either opt_g or opt_d, passed by the FL runtime
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.

    For GANs, this function will train either the generator or the discriminator
    based on the 'train_component' key in the config.
    """
    device = next(model.parameters()).device
    
    # Ensure model is a GANModel instance
    if not isinstance(model, GANModel):
        raise TypeError("Expected model to be an instance of GANModel.")

    imgs, _ = batch # MNIST batch contains (image, label), we only need image
    imgs = imgs.to(device)

    batch_size = imgs.size(0)
    latent_dim = model.latent_dim

    # Labels for adversarial loss
    valid = torch.ones(batch_size, 1, device=device)
    fake = torch.zeros(batch_size, 1, device=device)

    train_component = config.get("train_component", "discriminator") # Default to discriminator

    if train_component == "generator":
        # Sample noise as generator input
        z = torch.randn(batch_size, latent_dim, device=device)

        # Generate a batch of images
        gen_imgs = model.generator(z)

        # Adversarial loss for generator
        # We want discriminator to classify fake images as real (valid)
        g_loss = adversarial_loss(model.discriminator(gen_imgs), valid)
        return g_loss

    elif train_component == "discriminator":
        # Measure discriminator's ability to classify real from generated samples

        # Loss for real images
        real_loss = adversarial_loss(model.discriminator(imgs), valid)

        # Loss for fake images
        z = torch.randn(batch_size, latent_dim, device=device)
        # Detach to prevent gradients from flowing to generator
        gen_imgs = model.generator(z).detach() 
        fake_loss = adversarial_loss(model.discriminator(gen_imgs), fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        return d_loss

    else:
        raise ValueError(f"Unknown train_component: {train_component}. Must be 'generator' or 'discriminator'.")
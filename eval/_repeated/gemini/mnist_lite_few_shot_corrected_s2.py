"""
Auto-generated FL client module for GAN.
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

Special considerations for GANs:
  - build_model returns a single nn.Module (GANWrapper) containing both Generator and Discriminator.
  - train_step uses the 'train_mode' key in 'config' to determine whether to train the
    Generator or Discriminator, and which loss to return.
  - The FL runtime is expected to call train_step twice per batch:
    1. For the Generator, with its optimizer and config={"train_mode": "generator"}.
    2. For the Discriminator, with its optimizer and config={"train_mode": "discriminator"}.
"""

import math
import os  # For data_path management
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Assume torchvision is available for this environment.
# In a real FL setup, this dependency needs to be managed for clients.
_TORCHVISION_AVAILABLE = True
if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST
else:
    # Synthetic fallback for testing without torchvision
    class SyntheticMNISTDataset(Dataset):
        def __init__(self, n=200, img_shape=(1, 28, 28), num_classes=10):
            self.n = n
            self.img_shape = img_shape
            self.num_classes = num_classes
            # Images in [-1, 1] range to match Tanh output
            self.data = (torch.rand(n, *img_shape) * 2 - 1).float()
            self.targets = torch.randint(0, num_classes, (n,), dtype=torch.long)

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]


class Generator(nn.Module):
    """
    Generator network from the original script.
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
    Discriminator network from the original script.
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


class GANWrapper(nn.Module):
    """
    A wrapper nn.Module that encapsulates both Generator and Discriminator
    to be compatible with the FL client module's `build_model` contract.
    """

    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        latent_dim: int = 100,
    ):
        super().__init__()
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)
        self.img_shape = img_shape
        self.latent_dim = latent_dim

    # Define a forward pass for the wrapper that calls the generator.
    # This can be used for evaluation or initial generation by the FL server.
    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)


# ── FL Interface ────────────────────────────────────────────────────────


def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the GANWrapper model (containing both Generator and Discriminator).
    """
    model_kwargs = config.get("model_kwargs", {})
    # Default hyperparameters from the original script
    img_shape = model_kwargs.get("img_shape", (1, 28, 28))
    latent_dim = model_kwargs.get("latent_dim", 100)
    return GANWrapper(img_shape=img_shape, latent_dim=latent_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a DataLoader for the MNIST dataset.
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory = local.get("pin_memory", True)

    data_path = config.get("data_path", "./data")  # Default MNIST data directory

    if _TORCHVISION_AVAILABLE:
        is_train = (split == "train")
        # The original LightningModule uses normalize to [0.5], [0.5] which maps to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = MNIST(
            root=data_path,
            train=is_train,
            transform=transform,
            download=True  # In a real FL scenario, data would be pre-downloaded
        )
    else:
        # Synthetic fallback for testing without torchvision
        img_shape = config.get("model_kwargs", {}).get("img_shape", (1, 28, 28))
        dataset = SyntheticMNISTDataset(
            n=config.get("n_samples", 200 if split == "train" else 50),
            img_shape=img_shape
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,  # This is an instance of GANWrapper
    batch: tuple | list,
    optimizer,  # This will be either the Generator's or Discriminator's optimizer
    config: dict,
) -> torch.Tensor:
    """
    Performs ONE forward pass for either the Generator or Discriminator.
    Returns the raw loss tensor WITH grad_fn attached.

    The 'config' dictionary MUST contain a 'train_mode' key ("generator" or "discriminator")
    to specify which part of the GAN is being trained in this step.
    """
    if not isinstance(model, GANWrapper):
        raise TypeError("Expected model to be an instance of GANWrapper.")

    device = next(model.parameters()).device
    imgs, _ = batch  # MNIST provides images and labels; labels are not used for GAN training

    imgs = imgs.to(device)  # Real images

    train_mode = config.get("train_mode")
    if train_mode not in ["generator", "discriminator"]:
        raise ValueError(f"train_mode must be 'generator' or 'discriminator', but got {train_mode}")

    # Sample noise vector
    latent_dim = model.latent_dim
    z = torch.randn(imgs.shape[0], latent_dim).to(device)

    # Labels for adversarial loss
    valid = torch.ones(imgs.size(0), 1).to(device)
    fake = torch.zeros(imgs.size(0), 1).to(device)

    if train_mode == "generator":
        # Train generator: maximize log(D(G(z)))
        # Set D to eval mode (no BN/Dropout updates)
        # The optimizer for G will only update G's parameters.
        model.discriminator.eval()
        model.generator.train()

        gen_imgs = model.generator(z)
        g_loss = model.adversarial_loss(model.discriminator(gen_imgs), valid)
        return g_loss

    elif train_mode == "discriminator":
        # Train discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        # Set G to eval mode (no BN/Dropout updates)
        # The optimizer for D will only update D's parameters.
        model.generator.eval()
        model.discriminator.train()

        # Measure discriminator's ability to classify real from generated samples
        # Real images loss
        real_loss = model.adversarial_loss(model.discriminator(imgs), valid)

        # Fake images loss
        # Detach generated images to stop gradients from flowing back to generator
        with torch.no_grad():
            gen_imgs = model.generator(z)
        fake_loss = model.adversarial_loss(model.discriminator(gen_imgs), fake)

        # Discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        return d_loss
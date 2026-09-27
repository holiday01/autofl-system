"""
Auto-generated FL client module for GAN (based on PyTorch Lightning example).
Original script: lightning/examples/generative_adversarial_net.py

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
  - A GAN typically has two models (Generator, Discriminator) and two optimizers.
  - The `build_model` function returns a single `nn.Module` (a wrapper containing both G and D).
  - The `train_step` expects a single `optimizer` and returns a single loss.
  - To handle GAN training, the `train_step` here branches based on `config["gan_step"]`
    ("generator" or "discriminator"). The FL runtime is expected to call `train_step`
    twice per batch, once for each role, passing the appropriate optimizer and config.
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Check for torchvision availability for MNIST dataset
try:
    import torchvision
    import torchvision.transforms as transforms
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False
    print("WARNING: torchvision not available. Synthetic data will be used for MNIST.", flush=True)


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


class GAN(nn.Module):  # Changed from LightningModule to nn.Module
    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        lr: float = 0.0002,  # Not used internally, but kept for config consistency
        b1: float = 0.5,
        b2: float = 0.999,
        latent_dim: int = 100,
    ):
        super().__init__()
        # Manually store parameters (replaces save_hyperparameters from LightningModule)
        self.img_shape = img_shape
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim

        # networks
        self.generator = Generator(latent_dim=self.latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        # validation_z for logging images, can be kept
        self.validation_z = torch.randn(8, self.latent_dim)

    def forward(self, z):
        # The GAN's forward pass is typically the generator's forward pass
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    # Defaults from original script
    img_shape = kwargs.get("img_shape", (1, 28, 28))
    latent_dim = kwargs.get("latent_dim", 100)
    # lr, b1, b2 are primarily for optimizers, but GAN class stores them.
    lr = kwargs.get("lr", 0.0002)
    b1 = kwargs.get("b1", 0.5)
    b2 = kwargs.get("b2", 0.999)
    return GAN(img_shape=img_shape, lr=lr, b1=b1, b2=b2, latent_dim=latent_dim)


class SyntheticMNISTDataset(Dataset):
    """
    Synthetic fallback for MNIST when torchvision is not available or for testing.
    Generates random images and labels.
    """
    def __init__(self, n: int = 200, img_shape: tuple = (1, 28, 28)):
        self.n = n
        self.img_shape = img_shape
        # Generate random images, scaling them roughly to [0, 1] for MNIST-like appearance
        self.images = torch.rand(n, *img_shape) * 2 - 1 # Scaled to [-1, 1] like actual data
        self.labels = torch.randint(0, 10, (n,)) # 10 classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 16))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory = local.get("pin_memory", True)
    data_path = config.get("data_path", "./data")

    if _TORCHVISION_AVAILABLE:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Scale pixel values to [-1, 1]
        ])
        is_train = split == "train"
        dataset = torchvision.datasets.MNIST(
            root=data_path,
            train=is_train,
            download=True,
            transform=transform,
        )
    else:
        print(f"WARNING: torchvision not available for split '{split}'. Using synthetic data.", flush=True)
        img_shape = config.get("model_kwargs", {}).get("img_shape", (1, 28, 28))
        dataset = SyntheticMNISTDataset(n=200, img_shape=img_shape)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,  # This 'optimizer' will be either opt_g or opt_d from the FL runtime
    config: dict,
) -> torch.Tensor:
    """
    Performs one sub-step of GAN training (either Generator or Discriminator).
    Returns the raw loss tensor WITH grad_fn attached.

    CONTRACT (read carefully):
      - Do NOT call loss.backward() inside train_step.
      - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
      - Do NOT call .item() on the returned loss.
      - The FL runtime owns backward(), step(), and metric extraction.

    Special Note for GANs:
      - This `train_step` performs forward passes for either the Generator or Discriminator
        based on `config["gan_step"]` ("generator" or "discriminator").
      - For Discriminator training, an intermediate `model(z).detach()` call (which invokes
        the generator) is made. This is crucial for the GAN algorithm to prevent gradient
        flow to the generator during discriminator updates. The *final returned loss tensor*
        is NOT detached.
    """
    device = next(model.parameters()).device
    if isinstance(batch, (list, tuple)):
        # For GANs, the target (label) is not used for direct loss calculation
        # but the inputs (images) are needed.
        inputs, _ = batch[0], batch[1]
        inputs = inputs.to(device)
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs = batch.get("input", batch.get("x", batch.get("image")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # Assume 'model' is the GAN wrapper
    assert isinstance(model, GAN), "Model must be an instance of the GAN wrapper class."

    # Sample noise
    z = torch.randn(inputs.shape[0], model.latent_dim, device=device)

    # Determine which GAN step to perform
    gan_step = config.get("gan_step", "generator")  # Default to generator step

    if gan_step == "generator":
        # Train generator
        # The FL runtime is expected to zero_grad for opt_g before this call
        # The FL runtime is expected to do backward() and step() for opt_g after this call
        valid = torch.ones(inputs.size(0), 1, device=device)
        g_loss = model.adversarial_loss(model.discriminator(model(z)), valid)
        return g_loss
    elif gan_step == "discriminator":
        # Train discriminator
        # The FL runtime is expected to zero_grad for opt_d before this call
        # The FL runtime is expected to do backward() and step() for opt_d after this call
        valid = torch.ones(inputs.size(0), 1, device=device)
        real_loss = model.adversarial_loss(model.discriminator(inputs), valid)

        fake = torch.zeros(inputs.size(0), 1, device=device)
        # Generate fake images without tracking gradients for the generator here,
        # as per standard GAN training for discriminator update.
        fake_images = model(z).detach()

        fake_loss = model.adversarial_loss(model.discriminator(fake_images), fake)

        d_loss = (real_loss + fake_loss) / 2
        return d_loss
    else:
        raise ValueError(
            f"Unknown gan_step in config: '{gan_step}'. Expected 'generator' or 'discriminator'."
        )
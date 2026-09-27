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
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- Helper for GAN adversarial loss ---
def adversarial_loss(y_hat, y):
    return F.binary_cross_entropy_with_logits(y_hat, y)

# --- Original Generator class ---
class Generator(nn.Module):
    """
    Generator network. The federated model in this FL client.
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

# --- Original Discriminator class ---
class Discriminator(nn.Module):
    """
    Discriminator network. Used locally by each client to train the Generator.
    Not federated in this FL client conversion.
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


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds the federated model (the Generator in this GAN setup).
    """
    kwargs = config.get("model_kwargs", {})
    latent_dim = kwargs.get("latent_dim", config.get("latent_dim", 100))
    img_shape = kwargs.get("img_shape", (1, 28, 28)) # Default for MNIST
    return Generator(latent_dim=latent_dim, img_shape=img_shape)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds the DataLoader for MNIST data using Lightning's MNISTDataModule.
    """
    # To use MNISTDataModule, we need to mock its expected 'args' Namespace
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 32))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    data_path = config.get("data_path", os.getcwd()) # Default data dir is current working directory

    # Create a Namespace object to mimic cli args for MNISTDataModule
    # This is a workaround for LightningDataModule's design.
    class MockArgs:
        def __init__(self, data_dir, batch_size, num_workers):
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.num_workers = num_workers
    mock_args = MockArgs(data_dir=data_path, batch_size=batch_size, num_workers=num_workers)

    # Import locally to avoid top-level Lightning dependency if not always needed.
    # Assumes lightning.pytorch.demos.mnist_datamodule is accessible in the FL environment.
    from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule

    dm = MNISTDataModule(mock_args)
    dm.prepare_data() # Downloads and sets up data if not already present
    dm.setup(stage="fit") # Prepares train/val datasets

    if split == "train":
        return dm.train_dataloader()
    elif split == "val":
        return dm.val_dataloader()
    else:
        raise ValueError(f"Unsupported split: {split}. Expected 'train' or 'val'.")


def train_step(
    model: nn.Module, # This is the Generator instance
    batch: tuple | list,
    optimizer, # This is the Generator's optimizer
    config: dict,
) -> torch.Tensor:
    """
    Performs one step of GAN training for the Generator.
    The Discriminator is instantiated and trained locally within this step.

    CONTRACT:
      - This function returns the raw Generator loss tensor WITH grad_fn attached.
      - The FL runtime will call `loss.backward()` and `optimizer.step()` for the Generator.
      - The local Discriminator's training (`backward()` and `step()`) is managed internally
        and does not interact with the federated `optimizer` or the returned `loss`.
    """
    device = next(model.parameters()).device
    imgs, _ = batch # MNIST batch contains images and labels, we only need images for GAN

    # Move data to device
    imgs = imgs.to(device)

    # Get hyperparameters from config (or use defaults from original script)
    latent_dim = config.get("latent_dim", 100)
    img_shape = config.get("img_shape", (1, 28, 28))
    lr = config.get("lr", 0.0002)
    b1 = config.get("b1", 0.5)
    b2 = config.get("b2", 0.999)

    # Sample noise for generator input
    z = torch.randn(imgs.shape[0], latent_dim).to(device)

    # Initialize a local Discriminator and its optimizer for this training step.
    # This Discriminator is NOT federated; its parameters are not aggregated.
    # For simplicity, it starts fresh in each call to train_step.
    # In a more advanced FL setup, its state might be loaded from a local checkpoint.
    discriminator = Discriminator(img_shape=img_shape).to(device)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Define labels for adversarial loss
    valid = torch.ones(imgs.size(0), 1).to(device) # Label for real images / Generator's goal
    fake = torch.zeros(imgs.size(0), 1).to(device)  # Label for fake images

    # --- Discriminator training (local, full step) ---
    # This part is fully contained within train_step and does NOT involve the federated `optimizer`
    # or the `loss` that will be returned to the FL runtime for aggregation.
    opt_d.zero_grad()

    # Loss for real images: discriminator tries to classify real images as real
    real_loss = adversarial_loss(discriminator(imgs), valid)

    # Loss for fake images: discriminator tries to classify generated images as fake
    # Detach the generator's output here to prevent gradients flowing to Generator during D training
    fake_imgs = model(z).detach()
    fake_loss = adversarial_loss(discriminator(fake_imgs), fake)

    # Total discriminator loss
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward() # Perform local backward pass for Discriminator
    opt_d.step() # Perform local optimizer step for Discriminator

    # --- Generator training (forward pass only, loss returned for FL) ---
    # The Generator's parameters are updated by the FL runtime using the federated `optimizer`.
    # Generator wants the discriminator to classify its fake images as real (valid).
    # Its gradients will be computed by the FL runtime from this returned loss.
    g_loss = adversarial_loss(discriminator(model(z)), valid)

    return g_loss
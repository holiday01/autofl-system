import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

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
    def __init__(self, img_shape: tuple):
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


# Wrapper Module to contain both Generator and Discriminator
# This replaces the LightningModule's structure for the FL client's model
class GANModule(nn.Module):
    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        latent_dim: int = 100,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        # networks
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

    def forward(self, z):
        # The 'forward' of the GANModule will be the generator's forward
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the GANModule.
    """
    model_kwargs = config.get("model_kwargs", {})
    
    # Default img_shape for MNIST, can be overridden by config
    img_shape = model_kwargs.get("img_shape", (1, 28, 28)) 
    latent_dim = model_kwargs.get("latent_dim", 100)

    return GANModule(img_shape=img_shape, latent_dim=latent_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    full_dataset = None

    if _TORCHVISION_AVAILABLE:
        try:
            # Use the train split of MNIST, then split it into train/val
            full_dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=True, # Always use the training part of MNIST for client-side train/val split
                download=True,
                transform=transform,
            )
        except Exception as e:
            print(f"Warning: Could not load MNIST dataset from {data_path}. Error: {e}")

    if full_dataset is None:
        if allow_synthetic_data:
            print("Falling back to synthetic data.")
            # Generate synthetic MNIST-like data: images (1, 28, 28) and labels (0-9)
            num_samples = 1000 if split == "train" else 200
            # Generated images in range [-1, 1] to match Normalized MNIST
            synthetic_images = torch.randn(num_samples, 1, 28, 28) * 0.5 + 0.5 
            synthetic_labels = torch.randint(0, 10, (num_samples,))
            full_dataset = TensorDataset(synthetic_images, synthetic_labels)
        else:
            raise FileNotFoundError(
                f"MNIST dataset not found at {data_path} and allow_synthetic_data is False."
            )

    # Split dataset into train and validation sets (e.g., 80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    if split == "train":
        dataset_to_load = train_dataset
    elif split == "val":
        dataset_to_load = val_dataset
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'val'.")

    return DataLoader(dataset_to_load, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: torch.nn.Module, batch, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step().
    """
    imgs, _ = batch
    
    # Move batch to model's device
    device = next(model.parameters()).device
    imgs = imgs.to(device)

    # config.get("model_kwargs", {}) should contain `latent_dim` used by GANModule
    latent_dim = config.get("model_kwargs", {}).get("latent_dim", 100)

    # Sample noise
    z = torch.randn(imgs.shape[0], latent_dim).to(device)

    # Ground truth labels for adversarial loss
    valid_ones = torch.ones(imgs.size(0), 1).to(device)
    valid_zeros = torch.zeros(imgs.size(0), 1).to(device)

    # --- Compute Generator Loss ---
    # The generator wants to fool the discriminator into thinking generated images are real.
    # We pass generated images through discriminator and compare to 'real' labels.
    g_loss = model.adversarial_loss(model.discriminator(model(z)), valid_ones)

    # --- Compute Discriminator Loss ---
    # Discriminator's ability to classify real from generated samples.
    # Real images: label as real (ones)
    real_loss = model.adversarial_loss(model.discriminator(imgs), valid_ones)

    # Fake images: label as fake (zeros)
    # Detach to prevent gradients from flowing back to the generator during D's loss calculation.
    # This ensures that the fake_loss only updates the discriminator.
    fake_imgs = model(z).detach() 
    fake_loss = model.adversarial_loss(model.discriminator(fake_imgs), valid_zeros)

    # Discriminator loss is the average of these two components
    d_loss = (real_loss + fake_loss) / 2

    # Return combined loss for the FL runtime to perform backward and step.
    # This implies a single optimizer updates both G and D parameters based on their combined loss.
    # This is a common adaptation for GANs in standard FL frameworks when a single `train_step` is expected.
    # The gradients for G and D parameters will be accumulated from their respective loss components.
    return g_loss + d_loss
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
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


# Custom model container for FL, wrapping Generator and Discriminator
class FL_GAN_Model(nn.Module):
    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)
        self.img_shape = img_shape
        self.latent_dim = latent_dim # Store for easy access

    def forward(self, z):
        # The original GAN.forward calls generator
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)


def build_model(config: dict) -> torch.nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    # Extract hyperparameters needed for FL_GAN_Model
    latent_dim = model_kwargs.get("latent_dim", 100)
    img_shape = model_kwargs.get("img_shape", (1, 28, 28)) # Default MNIST shape

    model = FL_GAN_Model(latent_dim=latent_dim, img_shape=img_shape)
    return model


class SyntheticMNIST(Dataset):
    def __init__(self, num_samples=1000, img_shape=(1, 28, 28), num_classes=10):
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.data = [torch.randn(img_shape) for _ in range(num_samples)]
        self.targets = [torch.randint(0, num_classes, (1,)).item() for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    try:
        # Check if data_path exists, if not try to download
        os.makedirs(data_path, exist_ok=True)
        full_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    except Exception as e:
        if allow_synthetic_data:
            print(f"Warning: Could not load real MNIST data from {data_path}. Falling back to synthetic data. Error: {e}")
            # Synthetic data should mimic MNIST shape and class for consistency
            img_shape = config.get("model_kwargs", {}).get("img_shape", (1, 28, 28))
            full_dataset = SyntheticMNIST(num_samples=1000, img_shape=img_shape, num_classes=10)
        else:
            raise FileNotFoundError(
                f"Real dataset not found at {data_path} and 'allow_synthetic_data' is False."
            ) from e

    # Use random_split to produce train/val subsets (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    if split == "train":
        dataset = train_dataset
    elif split == "val":
        dataset = val_dataset
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: FL_GAN_Model, batch, optimizer, config: dict) -> torch.Tensor:
    imgs, _ = batch
    device = next(model.parameters()).device # Get model's device
    imgs = imgs.to(device)

    latent_dim = model.latent_dim # Retrieve latent_dim from the model instance

    # Sample noise
    z = torch.randn(imgs.shape[0], latent_dim, device=device)

    # Labels for adversarial loss
    valid = torch.ones(imgs.size(0), 1, device=device)
    fake = torch.zeros(imgs.size(0), 1, device=device)

    # Determine which component to train based on config
    # The FL runtime is expected to pass `train_component` in the config
    # and provide the appropriate optimizer (e.g., for generator or discriminator parameters).
    train_component = config.get("train_component", "generator")

    if train_component == "generator":
        # Train generator: G tries to make D classify fake images as real
        # D(G(z)) should be classified as real
        g_loss = model.adversarial_loss(model.discriminator(model.generator(z)), valid)
        return g_loss
    elif train_component == "discriminator":
        # Train discriminator: D tries to classify real images as real, fake images as fake

        # Real loss: D(real_imgs) should be classified as real
        real_loss = model.adversarial_loss(model.discriminator(imgs), valid)

        # Fake loss: D(G(z)) should be classified as fake
        # Crucially, detach the generator output to only train the discriminator
        fake_imgs = model.generator(z).detach() # Detach to prevent gradients flowing to G
        fake_loss = model.adversarial_loss(model.discriminator(fake_imgs), fake)

        # Discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        return d_loss
    else:
        raise ValueError(f"Invalid 'train_component' in config: {train_component}. Must be 'generator' or 'discriminator'.")
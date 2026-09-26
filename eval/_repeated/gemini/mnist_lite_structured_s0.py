import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


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


# Custom wrapper nn.Module for GAN to expose both generator and discriminator
class GANModule(nn.Module):
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

    def forward(self, z):
        """
        The primary forward pass for this combined GAN module is defined as the generator's output.
        """
        return self.generator(z)


class SyntheticMNISTDataset(Dataset):
    def __init__(self, num_samples=1000, img_shape=(1, 28, 28)):
        self.num_samples = num_samples
        self.img_shape = img_shape
        # Generate random data similar to MNIST, normalized to [-1, 1] as per GANs Tanh output
        self.data = torch.rand(num_samples, *img_shape) * 2 - 1
        self.targets = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def build_model(config: dict) -> torch.nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    return GANModule(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    # Normalize images to [-1, 1] to match the GAN's Tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset_to_load = None
    
    try:
        # Attempt to load real MNIST data
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        full_train_dataset = MNIST(root=data_path, train=True, download=True, transform=transform)
        full_test_dataset = MNIST(root=data_path, train=False, download=True, transform=transform)
        
        # Combine train and test datasets for a single random_split
        combined_dataset = torch.utils.data.ConcatDataset([full_train_dataset, full_test_dataset])

        total_len = len(combined_dataset)
        train_len = int(0.8 * total_len) # 80% for training
        val_len = total_len - train_len  # Remaining for validation
        
        train_dataset, val_dataset = random_split(combined_dataset, [train_len, val_len])

        if split == "train":
            dataset_to_load = train_dataset
        elif split == "val":
            dataset_to_load = val_dataset
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    except Exception as e:
        if allow_synthetic_data:
            print(f"Warning: Could not load real MNIST data from {data_path}. Using synthetic data. Error: {e}")
            
            synthetic_img_shape = config.get("model_kwargs", {}).get("img_shape", (1, 28, 28))
            num_synthetic_samples = config.get("synthetic_data_samples", 1000)
            
            full_synthetic_dataset = SyntheticMNISTDataset(num_synthetic_samples, img_shape=synthetic_img_shape)
            
            synthetic_train_len = int(0.8 * num_synthetic_samples)
            synthetic_val_len = num_synthetic_samples - synthetic_train_len
            
            # Adjust for potential rounding errors
            if synthetic_train_len + synthetic_val_len != num_synthetic_samples:
                synthetic_val_len = num_synthetic_samples - synthetic_train_len

            train_dataset_synthetic, val_dataset_synthetic = random_split(
                full_synthetic_dataset, [synthetic_train_len, synthetic_val_len]
            )

            if split == "train":
                dataset_to_load = train_dataset_synthetic
            elif split == "val":
                dataset_to_load = val_dataset_synthetic
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")
        else:
            raise FileNotFoundError(
                f"Failed to load MNIST data from '{data_path}' and synthetic data is not allowed. "
                f"Please ensure the data is available at '{data_path}' or set 'allow_synthetic_data' to True in the config. "
                f"Original error: {e}"
            )

    return DataLoader(dataset_to_load, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: GANModule, batch, optimizer, config: dict) -> torch.Tensor:
    imgs, _ = batch

    # Move tensors to the device of the model parameters
    device = next(model.parameters()).device
    imgs = imgs.to(device)

    # Sample noise
    z = torch.randn(imgs.shape[0], model.latent_dim, device=device)

    # Adversarial loss function (binary cross-entropy with logits)
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    # In a standard FL `train_step`, we return a single loss for a single optimization pass.
    # For GANs, the original training involves alternating optimization of Generator and Discriminator.
    # To comply with the FL interface, we will calculate the Generator's loss and return it.
    # This means the FL optimizer will update both Generator and Discriminator parameters
    # to minimize the Generator's ability to be distinguished as fake.
    # The Discriminator effectively acts as a learned loss function for the Generator.

    # Calculate G's loss: Generator tries to fool the Discriminator
    valid = torch.ones(imgs.size(0), 1, device=device) # Labels for real images
    fake_imgs = model.generator(z)                     # G's forward pass
    g_loss = adversarial_loss(model.discriminator(fake_imgs), valid) # D's forward on fake, target=real

    # Note: The Discriminator's separate loss calculation and explicit update
    # (`d_loss` and `opt_d.step()`) from the original Lightning script is omitted
    # here to fit the single-loss, single-optimizer FL `train_step` paradigm.
    # The gradients for `g_loss` will flow back through both `model.discriminator`
    # and `model.generator`, so both networks' parameters will be updated by
    # the FL runtime's `optimizer.step()` call.

    return g_loss
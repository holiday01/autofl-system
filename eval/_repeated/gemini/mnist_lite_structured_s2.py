# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# Check if torchvision is available for MNIST
try:
    import torchvision
    import torchvision.transforms as transforms
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


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


# Combined GAN Model (torch.nn.Module equivalent of LightningModule's core)
class GANClientModel(nn.Module):
    """
    A PyTorch Module encapsulating the Generator and Discriminator.
    The `forward` method is defined to return the output of the generator.
    """
    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        self.lr = lr
        self.b1 = b1
        self.b2 = b2

    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the GANClientModel.
    """
    model_kwargs = config.get("model_kwargs", {})
    img_shape = model_kwargs.get("img_shape", (1, 28, 28))
    latent_dim = model_kwargs.get("latent_dim", 100)
    lr = model_kwargs.get("lr", 0.0002)
    b1 = model_kwargs.get("b1", 0.5)
    b2 = model_kwargs.get("b2", 0.999)

    model = GANClientModel(
        img_shape=img_shape,
        latent_dim=latent_dim,
        lr=lr,
        b1=b1,
        b2=b2
    )
    return model


class SyntheticMNIST(Dataset):
    """
    A synthetic dataset mimicking MNIST data shape and values.
    """
    def __init__(self, num_samples: int = 1000, img_shape: tuple = (1, 28, 28), num_classes: int = 10):
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = 2 * torch.rand(self.img_shape) - 1
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Returns a DataLoader for the requested split ("train" or "val").
    Includes synthetic data fallback.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    val_ratio = config.get("val_ratio", 0.2)
    seed = config.get("seed", 42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = None
    if _TORCHVISION_AVAILABLE and os.path.exists(data_path):
        try:
            full_dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=True,
                download=True,
                transform=transform,
            )
            dataset = full_dataset
        except Exception:
            if allow_synthetic_data:
                img_shape = config.get("model_kwargs", {}).get("img_shape", (1, 28, 28))
                dataset = SyntheticMNIST(num_samples=1000, img_shape=img_shape)
            else:
                raise FileNotFoundError(
                    f"MNIST dataset not found at {data_path} and allow_synthetic_data is False. "
                    "Set 'allow_synthetic_data: True' in config to use synthetic data."
                )
    elif allow_synthetic_data:
        img_shape = config.get("model_kwargs", {}).get("img_shape", (1, 28, 28))
        dataset = SyntheticMNIST(num_samples=1000, img_shape=img_shape)
    else:
        raise FileNotFoundError(
            f"MNIST dataset not found at {data_path} and allow_synthetic_data is False. "
            "Set 'allow_synthetic_data: True' in config to use synthetic data."
        )

    num_total = len(dataset)
    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [num_train, num_val], generator=generator
    )

    if split == "train":
        dataset_to_load = train_dataset
    elif split == "val":
        dataset_to_load = val_dataset
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'val'.")

    return DataLoader(dataset_to_load, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: GANClientModel, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass and return the loss tensor for GAN training.

    This function is adapted to the FL client module's `train_step` signature.
    It computes both generator and discriminator losses, but must return a single
    loss tensor. The 'train_component' in the config allows specifying which loss
    to return, enabling the FL orchestrator to control which part of the GAN
    (Generator or Discriminator) is targeted for gradient aggregation.

    Note: The original GAN training involves two adversarial optimization steps.
    This `train_step` as a single function cannot perform both updates while
    strictly adhering to "Do NOT call loss.backward() or optimizer.step()".
    It is assumed that the FL runtime will call `loss.backward()` and
    `optimizer.step()` based on the returned loss tensor, and `optimizer`
    will be configured to target the appropriate model component.
    """
    device = next(model.parameters()).device
    imgs, _ = batch
    imgs = imgs.to(device)

    valid = torch.ones(imgs.size(0), 1).to(device)
    fake = torch.zeros(imgs.size(0), 1).to(device)

    latent_dim = model.latent_dim
    z = torch.randn(imgs.shape[0], latent_dim).to(device)

    # --- Compute Discriminator Loss (D_loss) ---
    real_loss = model.adversarial_loss(model.discriminator(imgs), valid)
    fake_imgs = model.generator(z).detach()
    fake_loss = model.adversarial_loss(model.discriminator(fake_imgs), fake)
    d_loss = (real_loss + fake_loss) / 2

    # --- Compute Generator Loss (G_loss) ---
    g_loss = model.adversarial_loss(model.discriminator(model.generator(z)), valid)

    train_component = config.get("local", {}).get("train_component", "discriminator")

    if train_component == "generator":
        return g_loss
    elif train_component == "discriminator":
        return d_loss
    else:
        raise ValueError(
            f"Invalid 'train_component' in config: {train_component}. "
            "Expected 'generator' or 'discriminator'."
        )
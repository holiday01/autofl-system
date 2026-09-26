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
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
except ImportError:
    try:
        import torchvision as _tv  # noqa: F401
        _TORCHVISION_AVAILABLE = True
    except ImportError:
        _TORCHVISION_AVAILABLE = False

if _TORCHVISION_AVAILABLE:
    import torchvision
    import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Preserved model architecture (exact copies from original script)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FL-compatible wrapper: replaces the LightningModule (GAN) with a plain
# nn.Module that the FL runtime can treat as a single trainable model.
# ---------------------------------------------------------------------------

class GANModel(nn.Module):
    """Thin nn.Module wrapper around Generator + Discriminator.

    Replaces the original ``GAN`` LightningModule.  The FL runtime owns the
    optimiser and the backward pass; ``train_step`` returns a combined loss
    (g_loss + d_loss) whose gradients are valid for both sub-networks.
    """

    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        latent_dim: int = 100,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return GANModel.

    Args:
        config: FL config dict.  Reads ``config.get("model_kwargs", {})``.
                Supported keys inside model_kwargs: img_shape, latent_dim.
    """
    kwargs = config.get("model_kwargs", {})
    return GANModel(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for MNIST (train or val split).

    Args:
        config: FL config dict.
            - config["local"]["batch_size"]      (default 16)
            - config["data_path"]                (default ".")
            - config["allow_synthetic_data"]     (default False)
        split: "train" or "val".

    Raises:
        FileNotFoundError: if real data is unavailable and
            config["allow_synthetic_data"] is False (or absent).
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    # ---- attempt to load real MNIST ----------------------------------------
    real_dataset = None
    load_error: Exception | None = None

    if _TORCHVISION_AVAILABLE:
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            real_dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=True,
                download=True,
                transform=transform,
            )
        except Exception as exc:  # noqa: BLE001
            load_error = exc
    else:
        load_error = ImportError("torchvision is not installed")

    if real_dataset is not None:
        n_val = max(1, int(len(real_dataset) * 0.1))
        n_train = len(real_dataset) - n_val
        train_subset, val_subset = random_split(
            real_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_subset if split == "train" else val_subset
        return DataLoader(
            chosen,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=True,
        )

    # ---- real data unavailable: honour allow_synthetic_data flag -----------
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"MNIST dataset could not be loaded from '{data_path}' "
            f"(reason: {load_error}).  "
            "Set config['allow_synthetic_data'] = True to fall back to "
            "synthetic data for smoke-testing."
        )

    # ---- synthetic fallback (only reached when flag is True) ---------------
    img_shape = config.get("model_kwargs", {}).get("img_shape", (1, 28, 28))
    n_samples = 1000 if split == "train" else 200
    images = torch.randn(n_samples, *img_shape)
    labels = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=True,
    )


def train_step(
    model: GANModel,
    batch,
    optimizer,  # single FL-managed optimiser covering all parameters
    config: dict,
) -> torch.Tensor:
    """One forward pass returning a combined GAN loss (grad attached).

    The FL runtime calls loss.backward() and optimizer.step() — this function
    must NOT do so.

    Combined loss = g_loss + d_loss, computed so that:
      - generator gradients flow only from g_loss
        (fake images are *not* detached when scoring for the generator)
      - discriminator gradients flow from real_loss + fake_loss
        (fake images *are* detached to block generator gradients there)

    Args:
        model:     GANModel instance.
        batch:     (imgs, labels) tuple from the DataLoader.
        optimizer: provided by FL runtime (unused here per contract).
        config:    FL config dict (currently unused inside this function).

    Returns:
        Combined scalar loss tensor with requires_grad=True.
    """
    imgs, _ = batch
    device = next(model.parameters()).device
    imgs = imgs.to(device)

    batch_size = imgs.size(0)
    latent_dim = model.latent_dim

    # ---- sample latent noise -----------------------------------------------
    z = torch.randn(batch_size, latent_dim, device=device)

    # ---- generator loss: fool discriminator into predicting "real" ----------
    valid = torch.ones(batch_size, 1, device=device)
    fake_imgs = model.generator(z)                          # not detached → grad flows to G
    g_loss = model.adversarial_loss(model.discriminator(fake_imgs), valid)

    # ---- discriminator loss ------------------------------------------------
    # real branch
    real_loss = model.adversarial_loss(model.discriminator(imgs), valid)

    # fake branch: detach so D gradients do not back-prop into G here
    fake = torch.zeros(batch_size, 1, device=device)
    fake_loss = model.adversarial_loss(
        model.discriminator(fake_imgs.detach()), fake
    )
    d_loss = (real_loss + fake_loss) / 2

    # ---- combined loss (FL runtime owns backward + step) -------------------
    loss = g_loss + d_loss
    return loss
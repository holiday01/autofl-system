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
    import torchvision
    import torchvision.transforms as transforms
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


# ── Preserved model architecture ─────────────────────────────────────────────

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


class GANModel(nn.Module):
    """Combined nn.Module wrapping Generator + Discriminator for FL.

    Extracted from the original GAN LightningModule. The forward() pass
    delegates to the generator, matching the original GAN.forward() behaviour.
    Both sub-networks are named sub-modules so FL weight aggregation covers
    the full parameter set.
    """

    def __init__(self, img_shape: tuple = (1, 28, 28), latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

    def forward(self, z):
        return self.generator(z)


# ── FL interface ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the GANModel.

    Recognised model_kwargs
    -----------------------
    img_shape  : tuple  – default (1, 28, 28)
    latent_dim : int    – default 100
    """
    kwargs = config.get("model_kwargs", {})
    img_shape = tuple(kwargs.get("img_shape", (1, 28, 28)))
    latent_dim = int(kwargs.get("latent_dim", 100))
    return GANModel(img_shape=img_shape, latent_dim=latent_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for MNIST 'train' or 'val' split.

    The full training set is split 80 / 20 (train / val) using random_split
    with a fixed seed for reproducibility across FL clients.

    If MNIST is unavailable and config['allow_synthetic_data'] is True, a
    synthetic TensorDataset (torch.randn images, randint labels) is used.
    If the flag is False (default), FileNotFoundError is raised instead.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    full_dataset = None

    if _TORCHVISION_AVAILABLE:
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            full_dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=True,
                download=False,
                transform=transform,
            )
        except Exception as exc:
            if not config.get("allow_synthetic_data", False):
                raise FileNotFoundError(
                    f"MNIST dataset not found at '{data_path}'. "
                    "Download it first (torchvision.datasets.MNIST(..., download=True)) "
                    "or set config['allow_synthetic_data'] = True to use a synthetic "
                    "stand-in for testing."
                ) from exc
            # Synthetic fallback – gated on allow_synthetic_data above
            n = 1000
            full_dataset = TensorDataset(
                torch.randn(n, 1, 28, 28),
                torch.randint(0, 10, (n,)),
            )
    else:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                "torchvision is not installed so MNIST cannot be loaded. "
                "Install torchvision or set config['allow_synthetic_data'] = True "
                "to use a synthetic stand-in for testing."
            )
        # Synthetic fallback – gated on allow_synthetic_data above
        n = 1000
        full_dataset = TensorDataset(
            torch.randn(n, 1, 28, 28),
            torch.randint(0, 10, (n,)),
        )

    n_total = len(full_dataset)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        full_dataset,
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


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """One GAN forward pass; returns the combined loss WITH grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step() – this function does neither.

    Loss composition
    ----------------
    The original LightningModule uses two separate optimisers and calls
    manual_backward twice per step.  In the FL setting a single combined loss
    is returned so that a single backward pass propagates gradients correctly
    into both sub-networks:

      loss = g_loss + d_loss

    Gradient flow
    -------------
    * g_loss   – discriminator(generator(z)) vs. all-real labels.
                 Gradients flow through BOTH discriminator and generator.
    * d_loss   – (real_loss + fake_loss) / 2.
                 fake_loss uses fake_imgs.detach(), so NO gradient reaches
                 the generator through d_loss — matching the original logic.
    """
    imgs, _ = batch
    device = next(model.parameters()).device
    imgs = imgs.to(device)

    batch_size = imgs.size(0)
    latent_dim = model.latent_dim

    # ── Sample noise ──────────────────────────────────────────────────────────
    z = torch.randn(batch_size, latent_dim, device=device)

    # ── Generator loss ────────────────────────────────────────────────────────
    # Generator wants discriminator to label its output as real (label = 1)
    fake_imgs = model.generator(z)                          # keep grad graph
    valid = torch.ones(batch_size, 1, device=device)
    g_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs), valid
    )

    # ── Discriminator loss ────────────────────────────────────────────────────
    real_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(imgs), valid
    )
    fake = torch.zeros(batch_size, 1, device=device)
    fake_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs.detach()),            # stop gen gradient
        fake,
    )
    d_loss = (real_loss + fake_loss) / 2

    # ── Combined loss (grad attached; backward/step handled by FL runtime) ────
    loss = g_loss + d_loss
    return loss
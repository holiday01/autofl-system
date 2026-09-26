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


# ---------------------------------------------------------------------------
# Original model architecture — preserved exactly
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
# Thin wrapper that combines Generator + Discriminator into one nn.Module
# so the FL runtime can treat the GAN as a single model object.
# ---------------------------------------------------------------------------

class GANModel(nn.Module):
    """Joint Generator + Discriminator module exposed to the FL runtime."""

    def __init__(self, img_shape: tuple = (1, 28, 28), latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

    def forward(self, z):
        """Generate images from a latent vector (mirrors the original GAN.forward)."""
        return self.generator(z)


# ---------------------------------------------------------------------------
# FL interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the GANModel.

    Recognised model_kwargs
    -----------------------
    img_shape  : tuple, default (1, 28, 28)
    latent_dim : int,   default 100
    """
    kwargs = config.get("model_kwargs", {})
    img_shape = tuple(kwargs.get("img_shape", (1, 28, 28)))
    latent_dim = int(kwargs.get("latent_dim", 100))
    return GANModel(img_shape=img_shape, latent_dim=latent_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Real data
    ---------
    Looks for an MNIST dataset under ``config['data_path']`` (default ".").
    The dataset is **not** re-downloaded automatically (``download=False``);
    place the MNIST files under ``<data_path>/MNIST/`` in advance.

    Synthetic fallback
    ------------------
    Only activated when ``config['allow_synthetic_data'] is True``.
    If the real data is missing and the flag is False, raises FileNotFoundError.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    img_shape = tuple(config.get("model_kwargs", {}).get("img_shape", (1, 28, 28)))

    # ---- attempt real MNIST -----------------------------------------------
    if _TORCHVISION_AVAILABLE:
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            full_ds = torchvision.datasets.MNIST(
                root=data_path,
                train=True,
                download=False,
                transform=transform,
            )
            n_total = len(full_ds)
            n_val = max(1, int(0.1 * n_total))
            n_train = n_total - n_val
            train_ds, val_ds = random_split(
                full_ds,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            chosen = train_ds if split == "train" else val_ds
            return DataLoader(
                chosen,
                batch_size=batch_size,
                shuffle=(split == "train"),
                drop_last=True,
            )
        except Exception:
            # Dataset files absent — fall through to synthetic fallback check
            pass

    # ---- synthetic fallback (must be explicitly opted-in) -----------------
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"MNIST dataset not found at '{data_path}'. "
            "Download it first (torchvision.datasets.MNIST(..., download=True)) "
            "or set config['allow_synthetic_data'] = True to use synthetic data."
        )

    n_samples = 1000 if split == "train" else 200
    x = torch.randn(n_samples, *img_shape)
    y = torch.randint(0, 10, (n_samples,))
    synthetic_ds = TensorDataset(x, y)
    return DataLoader(
        synthetic_ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=True,
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,            # accepted for API compatibility; FL runtime owns it
    config: dict,
) -> torch.Tensor:
    """One forward pass through the GAN; returns a combined loss WITH grad.

    Loss = generator_loss + discriminator_loss

    The generator loss encourages fake images to be classified as real.
    The discriminator loss measures its ability to separate real from fake.

    Neither loss.backward() nor optimizer.step() is called here — the FL
    runtime is responsible for both.
    """
    device = next(model.parameters()).device

    imgs, _ = batch
    imgs = imgs.to(device)
    batch_size = imgs.size(0)
    latent_dim = model.latent_dim

    # ---- sample latent noise ----------------------------------------------
    z = torch.randn(batch_size, latent_dim, device=device)

    # ---- generator loss ---------------------------------------------------
    # Ground-truth labels for "real" (all ones) placed on the correct device
    valid = torch.ones(batch_size, 1, device=device)
    fake_imgs = model.generator(z)
    g_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs), valid
    )

    # ---- discriminator loss -----------------------------------------------
    # Real images should be labelled real
    real_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(imgs), valid
    )
    # Fake images should be labelled fake (detach so grad doesn't flow to G here)
    fake_labels = torch.zeros(batch_size, 1, device=device)
    fake_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs.detach()), fake_labels
    )
    d_loss = (real_loss + fake_loss) / 2

    # ---- combined loss (grad attached, no .backward() called) -------------
    loss = g_loss + d_loss
    return loss
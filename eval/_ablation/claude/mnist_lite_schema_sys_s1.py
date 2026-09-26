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

# Lightning and torchvision are kept as optional so the FL client can be
# imported in environments that only have plain PyTorch installed.
_TORCHVISION_AVAILABLE = False
try:
    from lightning.pytorch import cli_lightning_logo
    from lightning.pytorch.core import LightningModule
    from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
    from lightning.pytorch.trainer import Trainer
    from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
except ImportError:
    pass

if _TORCHVISION_AVAILABLE:
    import torchvision
    import torchvision.transforms as transforms


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
# FL container — wraps Generator + Discriminator into a single nn.Module so
# that the FL runtime can treat them as one model for aggregation purposes.
# The original LightningModule (GAN) is intentionally not kept: its Trainer-
# specific hooks (manual_backward, toggle_optimizer, log_dict, …) have no
# meaning outside Lightning and would prevent the module from being imported
# in a plain-PyTorch FL environment.
# ---------------------------------------------------------------------------

class GANModel(nn.Module):
    """Single nn.Module container for FL that holds both sub-networks."""

    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

    def forward(self, z):
        return self.generator(z)


# ---------------------------------------------------------------------------
# FL interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the GAN model.

    Reads constructor arguments from ``config.get("model_kwargs", {})``:
      - img_shape  (tuple, default (1, 28, 28))
      - latent_dim (int,   default 100)
    """
    kwargs = config.get("model_kwargs", {})
    img_shape = tuple(kwargs.get("img_shape", (1, 28, 28)))
    latent_dim = int(kwargs.get("latent_dim", 100))
    return GANModel(latent_dim=latent_dim, img_shape=img_shape)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split (``"train"`` or ``"val"``).

    Data source priority
    --------------------
    1. MNIST loaded from ``config["data_path"]`` via torchvision (no download).
    2. Synthetic tensors — **only** when ``config["allow_synthetic_data"]``
       is ``True``.  When that flag is absent or ``False`` and the real dataset
       cannot be found, a ``FileNotFoundError`` is raised immediately so that
       the client never silently trains on random noise.

    Config keys read
    ----------------
    config["local"]["batch_size"]   → DataLoader batch size  (default 16)
    config["data_path"]             → root directory for MNIST (default ".")
    config["allow_synthetic_data"]  → synthetic fallback gate (default False)
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    dataset = None

    if _TORCHVISION_AVAILABLE:
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=True,
                download=False,
                transform=transform,
            )
        except Exception:
            dataset = None

    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset not found at '{data_path}'. "
                "Either place the dataset there (torchvision MNIST layout) or "
                "set config['allow_synthetic_data'] = True to use synthetic "
                "random data for smoke-testing purposes only."
            )
        # Synthetic fallback: (N, 1, 28, 28) images in [-1, 1], integer labels.
        n_samples = 1000
        imgs = torch.randn(n_samples, 1, 28, 28)
        labels = torch.randint(0, 10, (n_samples,))
        dataset = TensorDataset(imgs, labels)

    total = len(dataset)
    val_size = max(1, int(0.1 * total))
    train_size = total - val_size
    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
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
    """Run one forward pass and return the combined GAN loss with grad attached.

    The FL runtime is solely responsible for calling ``loss.backward()`` and
    ``optimizer.step()`` — this function must NOT do either.

    Because the FL runtime supplies a single optimizer that covers all model
    parameters, the two adversarial objectives are computed and summed:

    * **g_loss** — generator tries to fool the discriminator (gradient flows
      through both Generator and Discriminator).
    * **d_loss** — discriminator learns to separate real from generated images.
      The fake-image term uses ``fake_imgs.detach()`` so that the
      discriminator's classification gradient does not propagate back into the
      Generator via this path.

    Combining the losses gives each sub-network a meaningful gradient signal
    while keeping the interface compatible with a single shared optimizer.
    """
    device = next(model.parameters()).device

    imgs, _ = batch
    imgs = imgs.to(device)

    batch_size = imgs.size(0)
    latent_dim = model.latent_dim

    # ---- sample noise ----
    z = torch.randn(batch_size, latent_dim, device=device)

    # ---- generator loss: encourage generator to produce "real-looking" fakes ----
    fake_imgs = model.generator(z)
    valid = torch.ones(batch_size, 1, device=device)
    g_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs), valid
    )

    # ---- discriminator loss ----
    # real images → label 1
    real_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(imgs), valid
    )
    # generated images → label 0; detach so the discriminator's fake-image
    # gradient does not flow back through the generator
    fake_labels = torch.zeros(batch_size, 1, device=device)
    fake_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs.detach()), fake_labels
    )
    d_loss = (real_loss + fake_loss) / 2

    # combined loss; grad is attached — FL runtime calls backward() + step()
    loss = g_loss + d_loss
    return loss
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
# FL-compatible wrapper (replaces LightningModule)
# ---------------------------------------------------------------------------

class GANModule(nn.Module):
    """Plain nn.Module wrapping Generator + Discriminator for FL use."""

    def __init__(self, img_shape: tuple = (1, 28, 28), latent_dim: int = 100):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
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

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the GANModule.

    Recognised model_kwargs
    -----------------------
    img_shape  : tuple  (default (1, 28, 28))
    latent_dim : int    (default 100)
    """
    kwargs = config.get("model_kwargs", {})
    img_shape = tuple(kwargs.get("img_shape", (1, 28, 28)))
    latent_dim = int(kwargs.get("latent_dim", 100))
    return GANModule(img_shape=img_shape, latent_dim=latent_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for MNIST (train or val split).

    Config keys
    -----------
    local.batch_size          : int   (default 16)
    data_path                 : str   (default ".")
    allow_synthetic_data      : bool  (default False)
    model_kwargs.img_shape    : tuple used by synthetic fallback
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
                "Either provide a valid data_path containing a downloaded MNIST "
                "dataset, or set config['allow_synthetic_data'] = True to use "
                "randomly generated tensors for smoke-testing."
            )
        # Synthetic fallback — gated on allow_synthetic_data
        n_samples = 1000
        img_shape = tuple(config.get("model_kwargs", {}).get("img_shape", (1, 28, 28)))
        imgs = torch.randn(n_samples, *img_shape)
        labels = torch.randint(0, 10, (n_samples,))
        dataset = TensorDataset(imgs, labels)

    # Deterministic 90 / 10 train-val split
    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_subset, val_subset = random_split(
        dataset,
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
    model: nn.Module,
    batch,
    optimizer,          # provided by FL runtime; present for API compliance
    config: dict,
) -> torch.Tensor:
    """One forward pass through both G and D; returns a combined loss tensor.

    Generator loss  : fool the discriminator (BCE vs. all-ones labels).
    Discriminator loss : average of real-image loss and fake-image loss.

    The combined loss (g_loss + d_loss) lets the FL runtime call a single
    backward() pass that accumulates gradients for both sub-networks.
    Note: loss.backward() and optimizer.step() are intentionally omitted —
    the FL runtime is responsible for those calls.
    """
    device = next(model.parameters()).device

    imgs, _ = batch
    imgs = imgs.to(device)
    batch_size = imgs.size(0)

    latent_dim = model.latent_dim

    # ---- Sample noise -------------------------------------------------------
    z = torch.randn(batch_size, latent_dim, device=device)

    # ---- Generator loss: G(z) should fool D ---------------------------------
    generated = model.generator(z)                          # keep graph for G
    valid = torch.ones(batch_size, 1, device=device)
    g_loss = model.adversarial_loss(model.discriminator(generated), valid)

    # ---- Discriminator loss -------------------------------------------------
    # Real images → labelled 1
    real_loss = model.adversarial_loss(
        model.discriminator(imgs),
        torch.ones(batch_size, 1, device=device),
    )
    # Fake images → labelled 0  (detach so G gets no gradient from this path)
    fake_loss = model.adversarial_loss(
        model.discriminator(generated.detach()),
        torch.zeros(batch_size, 1, device=device),
    )
    d_loss = (real_loss + fake_loss) / 2

    # ---- Combined loss (has grad_fn; FL runtime owns backward + step) -------
    loss = g_loss + d_loss
    return loss
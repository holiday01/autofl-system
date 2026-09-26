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
    _TORCHVISION_AVAILABLE = False

if _TORCHVISION_AVAILABLE:
    import torchvision
else:
    try:
        import torchvision
        _TORCHVISION_AVAILABLE = True
    except ImportError:
        pass


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
    """Combined GAN module wrapping Generator and Discriminator for FL.

    Extracted from the GAN LightningModule; the LightningModule scaffolding
    (manual optimizers, logging, validation_z) is intentionally omitted — the
    FL runtime owns the training loop.
    """

    def __init__(self, img_shape: tuple = (1, 28, 28), latent_dim: int = 100):
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
    """Instantiate and return the GANModel.

    Recognised model_kwargs:
      img_shape  (tuple, default (1, 28, 28))
      latent_dim (int,   default 100)
    """
    kwargs = config.get("model_kwargs", {})
    img_shape = tuple(kwargs.get("img_shape", (1, 28, 28)))
    latent_dim = int(kwargs.get("latent_dim", 100))
    return GANModel(img_shape=img_shape, latent_dim=latent_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ("train" or "val").

    Attempts to load MNIST from *data_path*.  Falls back to synthetic tensors
    only when config["allow_synthetic_data"] is True; otherwise raises
    FileNotFoundError so the caller knows real data is missing.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    dataset = None

    if _TORCHVISION_AVAILABLE:
        try:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ])
            full_dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=True,
                download=True,
                transform=transform,
            )
            n_total = len(full_dataset)
            n_val = max(1, int(0.1 * n_total))
            n_train = n_total - n_val
            train_subset, val_subset = random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            dataset = train_subset if split == "train" else val_subset
        except Exception:
            dataset = None

    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset could not be loaded from '{data_path}' "
                "(torchvision unavailable or download failed). "
                "Set config['allow_synthetic_data'] = True to use synthetic "
                "data instead, or supply a valid 'data_path'."
            )
        # Synthetic fallback — gated on allow_synthetic_data flag above
        n_samples = 1000
        images = torch.randn(n_samples, 1, 28, 28)
        labels = torch.randint(0, 10, (n_samples,))
        full_synth = TensorDataset(images, labels)
        n_val = max(1, int(0.1 * n_samples))
        n_train = n_samples - n_val
        train_subset, val_subset = random_split(
            full_synth,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        dataset = train_subset if split == "train" else val_subset

    return DataLoader(
        dataset,
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
    """One forward pass over a GAN mini-batch.

    Returns the combined generator + discriminator loss WITH gradients
    attached.  The FL runtime is responsible for calling loss.backward()
    and optimizer.step(); this function must NOT do either.

    Combined loss breakdown
    -----------------------
    g_loss   : generator tries to fool the discriminator (fake → real labels)
    d_loss   : discriminator classifies real images as real AND
               detached fake images as fake; averaged.
    total    : g_loss + d_loss  (single backward pass updates both G and D)
    """
    imgs, _ = batch
    device = next(model.parameters()).device
    imgs = imgs.to(device)

    latent_dim = model.latent_dim

    # Sample latent noise once; reuse generated images for both losses
    z = torch.randn(imgs.size(0), latent_dim, device=device)
    fake_imgs = model.generator(z)                          # (B, C, H, W)

    valid = torch.ones(imgs.size(0), 1, device=device)

    # --- Generator loss: make discriminator believe fakes are real ---
    g_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs), valid
    )

    # --- Discriminator loss ---
    real_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(imgs), valid
    )
    fake_labels = torch.zeros(imgs.size(0), 1, device=device)
    # detach: discriminator update must not propagate through generator here
    fake_loss = F.binary_cross_entropy_with_logits(
        model.discriminator(fake_imgs.detach()), fake_labels
    )
    d_loss = (real_loss + fake_loss) / 2

    # Return combined loss; FL runtime calls .backward() and optimizer.step()
    return g_loss + d_loss
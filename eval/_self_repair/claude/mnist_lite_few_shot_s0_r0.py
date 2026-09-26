"""
FL client for mnist_lite.py (Lightning GAN) — few-shot style.
Correct FL contract: no backward/step in train_step, synthetic fallback included.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


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
        return self.model(img.view(img.size(0), -1))


class GANModel(nn.Module):
    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.generator = Generator(latent_dim, img_shape)
        self.discriminator = Discriminator(img_shape)
        self.latent_dim = latent_dim

    def forward(self, z):
        return self.generator(z)


def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return GANModel(
        latent_dim=kwargs.get("latent_dim", 100),
        img_shape=tuple(kwargs.get("img_shape", [1, 28, 28])),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    try:
        from torchvision import datasets, transforms
        full_ds = datasets.MNIST(
            data_path, train=True, download=True, transform=transforms.ToTensor()
        )
        n_val = max(1, int(0.1 * len(full_ds)))
        train_ds, val_ds = random_split(
            full_ds, [len(full_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_ds if split == "train" else val_ds
    except Exception:
        n = 1000
        data = torch.randn(n, 1, 28, 28)
        targets = torch.zeros(n, dtype=torch.long)
        full_ds = TensorDataset(data, targets)
        n_val = max(1, int(0.1 * n))
        train_ds, val_ds = random_split(
            full_ds, [n - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_ds if split == "train" else val_ds

    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    """One GAN forward pass — returns combined D+G loss with grad_fn attached.
    The FL runtime is responsible for loss.backward() and optimizer.step().
    """
    device = next(model.parameters()).device
    imgs = batch[0].to(device)
    batch_size = imgs.size(0)

    valid = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)

    z = torch.randn(batch_size, model.latent_dim, device=device)
    fake_imgs = model.generator(z)

    d_real = F.binary_cross_entropy_with_logits(model.discriminator(imgs), valid)
    d_fake = F.binary_cross_entropy_with_logits(model.discriminator(fake_imgs.detach()), fake_labels)
    g_loss = F.binary_cross_entropy_with_logits(model.discriminator(fake_imgs), valid)

    return d_real + d_fake + g_loss

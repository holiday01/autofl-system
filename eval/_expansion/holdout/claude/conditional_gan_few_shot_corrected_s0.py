"""
Auto-generated FL client module.
Original script: conditional_gan_keras.py (Keras/TF → PyTorch port)

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE generator forward pass and returns the raw generator
    loss tensor WITH grad_fn attached.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step for
    the generator optimizer (`opt`).
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step() on `opt`, and metric extraction.

  NOTE (GAN deviation): GANs require two optimizers. The discriminator is updated
  internally in train_step using a cached Adam optimizer stored on the model as
  `_d_optimizer`. Only the generator loss is returned to the FL runtime, which
  drives the generator optimizer passed in via `opt`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms


# ── Model Definitions ────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    def __init__(self, num_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        in_channels = num_channels + num_classes
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, _, H, W = images.shape
        label_map = F.one_hot(labels, self.num_classes).float()
        label_map = label_map.view(B, self.num_classes, 1, 1).expand(B, -1, H, W)
        x = torch.cat([images, label_map], dim=1)
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        in_dim = latent_dim + num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 7 * 7 * in_dim),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (in_dim, 7, 7)),
            nn.ConvTranspose2d(in_dim, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = F.one_hot(labels, self.num_classes).float()
        x = torch.cat([noise, label_emb], dim=1)
        return self.net(x)


class ConditionalGAN(nn.Module):
    """Wraps generator and discriminator; discriminator optimizer is cached internally."""

    def __init__(self, latent_dim: int = 128, num_classes: int = 10, num_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.generator = Generator(latent_dim=latent_dim, num_classes=num_classes)
        self.discriminator = Discriminator(num_channels=num_channels, num_classes=num_classes)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.generator(noise, labels)


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return ConditionalGAN(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    data_path = config.get("data_path", "./data")
    val_ratio = config.get("val_ratio", 0.1)

    transform = transforms.ToTensor()
    train_full = datasets.MNIST(root=data_path, train=True,  download=True, transform=transform)
    test_full  = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_full, test_full])

    n_val   = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
    )
    ds = train_ds if split == "train" else val_ds
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE generator forward pass.  Returns the raw generator loss WITH grad_fn attached.

    The FL runtime calls loss.backward() and optimizer.step() on the generator optimizer
    (`opt`) externally — do NOT do either here, and do NOT detach() or .item() the result.

    NOTE (GAN deviation): The discriminator step is handled internally. A discriminator
    Adam optimizer is created once and cached on `model._d_optimizer` to preserve
    momentum state across steps. Only the generator loss is returned to the FL runtime.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        real_images, labels = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        real_images = batch.get("image", batch.get("x", batch.get("input")))
        labels      = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    latent_dim = getattr(model, "latent_dim", config.get("model_kwargs", {}).get("latent_dim", 128))
    d_lr       = config.get("d_lr", config.get("lr", 3e-4))

    generator     = model.generator
    discriminator = model.discriminator
    loss_fn       = nn.BCEWithLogitsLoss()
    B             = real_images.size(0)

    if not hasattr(model, "_d_optimizer"):
        model._d_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999)
        )
    d_optimizer = model._d_optimizer

    # ── Discriminator step (internal) ────────────────────────────────────────
    d_optimizer.zero_grad()

    noise     = torch.randn(B, latent_dim, device=device)
    fake_imgs = generator(noise, labels).detach()

    real_preds = discriminator(real_images, labels)
    fake_preds = discriminator(fake_imgs,   labels)
    # Convention (mirrors original): real → 0, fake → 1
    d_loss = (
        loss_fn(real_preds, torch.zeros(B, 1, device=device))
        + loss_fn(fake_preds, torch.ones(B, 1, device=device))
    )
    d_loss.backward()
    d_optimizer.step()

    # ── Generator forward pass (returned to FL runtime) ──────────────────────
    noise     = torch.randn(B, latent_dim, device=device)
    fake_imgs = generator(noise, labels)
    fake_preds = discriminator(fake_imgs, labels)
    # Generator wants discriminator to classify fakes as real (→ 0 in this convention)
    g_loss = loss_fn(fake_preds, torch.zeros(B, 1, device=device))

    return g_loss
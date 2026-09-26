"""
Auto-generated FL client module.
Original script: DCGAN training script (torchvision DCGAN example)

Exposes:
  build_model(config)               -> nn.Module (GANWrapper: Generator + Discriminator)
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.

GAN NOTE:
  Generator and Discriminator are wrapped in GANWrapper so the FL runtime
  treats them as a single model.  train_step returns errD + errG computed in
  one combined forward pass; fake images are detached when computing errD so
  gradients reach G only through errG, matching the original alternating-update
  semantics as closely as a single backward pass allows.
"""
import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64, nc: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


class GANWrapper(nn.Module):
    """Wraps Generator + Discriminator as a single nn.Module for the FL runtime."""

    def __init__(self, nz: int = 100, ngf: int = 64, ndf: int = 64, nc: int = 3):
        super().__init__()
        self.nz = nz
        self.netG = Generator(nz=nz, ngf=ngf, nc=nc)
        self.netD = Discriminator(ndf=ndf, nc=nc)
        self.apply(weights_init)

    def forward(self, real_images: torch.Tensor) -> torch.Tensor:
        """Returns errD + errG with grad_fn. Called only from train_step."""
        device = real_images.device
        batch_size = real_images.size(0)
        criterion = nn.BCELoss()

        real_label = torch.ones(batch_size, dtype=real_images.dtype, device=device)
        fake_label = torch.zeros(batch_size, dtype=real_images.dtype, device=device)

        # D loss: real half
        errD_real = criterion(self.netD(real_images), real_label)

        # D loss: fake half — detach so D grad does not flow through G
        noise = torch.randn(batch_size, self.nz, 1, 1, device=device)
        fake = self.netG(noise)
        errD_fake = criterion(self.netD(fake.detach()), fake_label)

        errD = errD_real + errD_fake

        # G loss — keep fake attached so grad flows back to G
        errG = criterion(self.netD(fake), real_label)

        return errD + errG


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return GANWrapper(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local        = config.get("local", {})
    batch_size   = local.get("batch_size",  config.get("batch_size",  64))
    num_workers  = local.get("num_workers", config.get("num_workers", 2))
    pin_memory   = local.get("pin_memory",  True)

    dataset_type = config.get("dataset",    "cifar10")
    data_path    = config.get("data_path",  ".")
    image_size   = config.get("image_size", 64)

    dataset = None

    if dataset_type in ("imagenet", "folder", "lfw"):
        if os.path.isdir(data_path):
            dataset = dset.ImageFolder(
                root=data_path,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            )

    elif dataset_type == "lsun":
        classes = [c + "_train" for c in config.get("lsun_classes", "bedroom").split(",")]
        if os.path.isdir(data_path):
            dataset = dset.LSUN(
                root=data_path,
                classes=classes,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            )

    elif dataset_type == "cifar10":
        try:
            dataset = dset.CIFAR10(
                root=data_path,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            )
        except Exception:
            dataset = None

    elif dataset_type == "mnist":
        try:
            dataset = dset.MNIST(
                root=data_path,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]),
            )
        except Exception:
            dataset = None

    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Dataset '{dataset_type}' not found at '{data_path}'. "
                "Set config['allow_synthetic_data']=True to use a synthetic fallback."
            )
        nc      = config.get("model_kwargs", {}).get("nc", 3)
        n_synth = config.get("synthetic_n", 1000)
        X = torch.randn(n_synth, nc, image_size, image_size)
        y = torch.zeros(n_synth, dtype=torch.long)
        dataset = TensorDataset(X, y)

    val_ratio = config.get("val_ratio", 0.1)
    n_val     = max(1, int(len(dataset) * val_ratio))
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
    )
    ds = train_ds if split == "train" else val_ds
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass through GANWrapper.  Returns errD + errG WITH grad_fn.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        real_images = batch[0].to(device)
    elif isinstance(batch, dict):
        key = next(k for k in ("image", "x", "input") if k in batch)
        real_images = batch[key].to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    return model(real_images)
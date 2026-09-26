"""
Auto-generated FL client module.
Original script: DCGAN training script.

Exposes:
  build_model(config)               -> nn.Module (GANBundle containing Generator + Discriminator)
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT NOTES for GAN training:
  - build_model returns a GANBundle(netG, netD). The FL runtime federates netG weights;
    netD is a local, non-federated component.
  - train_step performs the full Discriminator update internally (backward + step for netD
    happen inside train_step because netD is client-local only).
  - train_step returns the Generator loss WITH grad_fn attached.
  - The passed-in optimizer (opt) must cover netG parameters only.
  - The FL runtime owns backward() and step() for netG — do NOT call either on the
    returned errG or the passed-in optimizer inside train_step.
  - Do NOT call .item() or .detach() on the returned loss.
"""
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
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
        if (x.is_cuda or x.is_xpu) and self.ngpu > 1:
            return nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
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
        if (x.is_cuda or x.is_xpu) and self.ngpu > 1:
            out = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            out = self.main(x)
        return out.view(-1, 1).squeeze(1)


class GANBundle(nn.Module):
    """Wraps Generator and Discriminator. FL runtime federates netG; netD stays local."""

    def __init__(self, netG: Generator, netD: Discriminator):
        super().__init__()
        self.netG = netG
        self.netD = netD

    def forward(self, x):
        return self.netG(x)


# ── FL Interface ─────────────────────────────────────────────────────────────


def build_model(config: dict) -> nn.Module:
    nc = config.get("nc", 3)
    g_kwargs = {"nc": nc, **config.get("generator_kwargs", {})}
    d_kwargs = {"nc": nc, **config.get("discriminator_kwargs", {})}

    netG = Generator(**g_kwargs)
    netD = Discriminator(**d_kwargs)
    netG.apply(weights_init)
    netD.apply(weights_init)

    checkpoint_g = config.get("netG_checkpoint", "")
    checkpoint_d = config.get("netD_checkpoint", "")
    if checkpoint_g:
        netG.load_state_dict(torch.load(checkpoint_g, map_location="cpu"))
    if checkpoint_d:
        netD.load_state_dict(torch.load(checkpoint_d, map_location="cpu"))

    return GANBundle(netG, netD)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    dataset_name = config.get("dataset", "fake").lower()
    dataroot     = config.get("dataroot", ".")
    image_size   = config.get("image_size", 64)
    nc           = config.get("nc", 3)

    norm_mean = (0.5,) * nc
    norm_std  = (0.5,) * nc
    center_crop_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    if dataset_name in ("imagenet", "folder", "lfw"):
        dataset = dset.ImageFolder(root=dataroot, transform=center_crop_transform)
    elif dataset_name == "lsun":
        classes = [c + "_train" for c in config.get("lsun_classes", "bedroom").split(",")]
        dataset = dset.LSUN(root=dataroot, classes=classes, transform=center_crop_transform)
    elif dataset_name == "cifar10":
        dataset = dset.CIFAR10(
            root=dataroot, download=True,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]),
        )
    elif dataset_name == "mnist":
        dataset = dset.MNIST(
            root=dataroot, download=True,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
    else:  # fake
        dataset = dset.FakeData(
            image_size=(3, image_size, image_size),
            transform=transforms.ToTensor(),
        )

    val_ratio = config.get("val_ratio", 0.1)
    n_val   = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
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
    Performs the full Discriminator update and one Generator forward pass.
    Returns errG WITH grad_fn attached.

    GAN-specific contract deviation:
      netD backward + step happen inside this function because netD is client-local.
      The passed-in optimizer covers netG only; the FL runtime calls backward() and
      step() on it after this function returns — do NOT do so here.
    """
    assert isinstance(model, GANBundle), "train_step requires a GANBundle"
    device = next(model.parameters()).device
    netG, netD = model.netG, model.netD

    # Lazily create and cache the D optimizer on the bundle object.
    if not hasattr(model, "_optimizerD"):
        lr    = config.get("lr", 2e-4)
        beta1 = config.get("beta1", 0.5)
        model._optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    optimizerD = model._optimizerD
    criterion  = nn.BCELoss()
    nz         = config.get("nz", 100)

    if isinstance(batch, (list, tuple)):
        real = batch[0].to(device)
    elif isinstance(batch, dict):
        real = batch.get("image", batch.get("x", batch.get("input"))).to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    b = real.size(0)
    real_label = torch.ones(b,  dtype=real.dtype, device=device)
    fake_label = torch.zeros(b, dtype=real.dtype, device=device)

    # ── Update D: maximize log(D(x)) + log(1 - D(G(z))) ─────────────────
    netD.zero_grad()
    errD_real = criterion(netD(real), real_label)
    errD_real.backward()

    noise     = torch.randn(b, nz, 1, 1, device=device)
    fake      = netG(noise)
    errD_fake = criterion(netD(fake.detach()), fake_label)
    errD_fake.backward()
    optimizerD.step()

    # ── Compute G loss — no backward; FL runtime owns that ────────────────
    noise = torch.randn(b, nz, 1, 1, device=device)
    fake  = netG(noise)
    errG  = criterion(netD(fake), real_label)
    return errG
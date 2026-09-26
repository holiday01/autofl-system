"""
Auto-generated FL client module.
Original script: dcgan_train.py

Exposes:
  build_model(config)                   -> nn.Module (GANModel)
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor
"""
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


# ── Original source (unchanged, minus top-level argparse/training loop) ────────


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if (input.is_cuda or input.is_xpu) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu=1, ndf=64, nc=3):
        super(Discriminator, self).__init__()
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
            nn.Sigmoid()
        )

    def forward(self, input):
        if (input.is_cuda or input.is_xpu) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class GANModel(nn.Module):
    """Wraps Generator and Discriminator as a single nn.Module for the FL harness."""
    def __init__(self, netG: Generator, netD: Discriminator):
        super().__init__()
        self.netG = netG
        self.netD = netD

    def forward(self, x):
        return self.netG(x)


# ── FL Interface ────────────────────────────────────────────────────────


def build_model(config: dict) -> nn.Module:
    """
    Instantiate Generator and Discriminator wrapped in GANModel.
    config['model_kwargs'] may include: ngpu, nz, ngf, ndf, nc, netG (path), netD (path).
    """
    kwargs = config.get("model_kwargs", {})
    ngpu = kwargs.get("ngpu", 1)
    nz   = kwargs.get("nz",   100)
    ngf  = kwargs.get("ngf",  64)
    ndf  = kwargs.get("ndf",  64)
    nc   = kwargs.get("nc",   3)

    netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
    netD = Discriminator(ngpu=ngpu, ndf=ndf, nc=nc)
    netG.apply(weights_init)
    netD.apply(weights_init)

    netG_path = kwargs.get("netG", "")
    netD_path = kwargs.get("netD", "")
    if netG_path:
        netG.load_state_dict(torch.load(netG_path))
    if netD_path:
        netD.load_state_dict(torch.load(netD_path))

    return GANModel(netG, netD)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for the requested split.
    config['dataset']   : cifar10 | mnist | fake | folder | lsun | lfw | imagenet
    config['data_path'] : root directory for the dataset
    config['local']     : may override batch_size, num_workers, pin_memory
    config['dataset_kwargs']['imageSize'] : spatial resolution (default 64)
    config['dataset_kwargs']['classes']   : LSUN class list (default 'bedroom')
    """
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory",  True)

    dataset_kwargs = config.get("dataset_kwargs", {})
    dataset_type   = config.get("dataset", "fake").lower()
    data_path      = config.get("data_path", ".")
    image_size     = dataset_kwargs.get("imageSize", config.get("imageSize", 64))

    if dataset_type in ("imagenet", "folder", "lfw"):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_dataset = dset.ImageFolder(root=data_path, transform=transform)

    elif dataset_type == "lsun":
        classes = [c + "_train" for c in dataset_kwargs.get("classes", "bedroom").split(",")]
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_dataset = dset.LSUN(root=data_path, classes=classes, transform=transform)

    elif dataset_type == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_dataset = dset.CIFAR10(root=data_path, download=True, transform=transform)

    elif dataset_type == "mnist":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        full_dataset = dset.MNIST(root=data_path, download=True, transform=transform)

    else:  # fake
        nc = config.get("model_kwargs", {}).get("nc", 3)
        full_dataset = dset.FakeData(
            image_size=(nc, image_size, image_size),
            transform=transforms.ToTensor(),
        )

    val_ratio = config.get("val_ratio", 0.1)
    n_val     = max(1, int(len(full_dataset) * val_ratio))
    n_train   = len(full_dataset) - n_val
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
    One GAN forward pass — returns combined D+G loss with grad_fn attached.
    The FL runtime is responsible for loss.backward() and optimizer.step().
    This function must NOT call either.
    """
    device = next(model.parameters()).device
    nz = config.get("model_kwargs", {}).get("nz", 100)
    criterion = nn.BCELoss()

    if isinstance(batch, (list, tuple)):
        real_images = batch[0].to(device)
    elif isinstance(batch, dict):
        real_images = batch.get("image", batch.get("x", batch.get("input"))).to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    batch_size = real_images.size(0)
    netD = model.netD
    netG = model.netG

    # Discriminator forward on real images
    real_label = torch.ones(batch_size, device=device)
    errD_real = criterion(netD(real_images), real_label)

    # Discriminator forward on generated images
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(noise)
    fake_label = torch.zeros(batch_size, device=device)
    errD_fake = criterion(netD(fake.detach()), fake_label)

    # Generator forward (discriminator sees fake as real)
    errG = criterion(netD(fake), real_label)

    # Return combined loss — grad_fn intact; FL runtime handles backward/step
    return errD_real + errD_fake + errG


def _get_scaler():
    if not hasattr(_get_scaler, "_scaler"):
        _get_scaler._scaler = torch.cuda.amp.GradScaler()
    return _get_scaler._scaler
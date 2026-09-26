"""
Auto-generated FL client module.
Original script: DCGAN training script

Exposes:
  build_model(config)                    -> nn.ModuleDict {'netG': Generator, 'netD': Discriminator}
  build_dataloader(config, split)        -> DataLoader
  train_step(model, batch, opt, config)  -> loss tensor (errG, with grad_fn)

GAN-SPECIFIC CONTRACT DEVIATION:
  Standard FL contract: train_step performs ONE forward pass, returns raw loss;
  the FL runtime owns backward() and optimizer.step().

  GANs require interleaved D and G updates that cannot be decomposed into a
  single forward pass + one backward.  This module adopts a split-responsibility
  convention:

    - The Discriminator is LOCAL state.  train_step performs the full D update
      (zero_grad → forward → backward → step) internally.
    - The Generator loss (errG) is returned WITH grad_fn attached so the FL
      runtime can call loss.backward() + optimizerG.step() externally.

  `optimizer` must be a dict: {'optimizerG': <optim>, 'optimizerD': <optim>}.
  `model`     must be an nn.ModuleDict returned by build_model().

  Do NOT call errG.backward() or .item() before passing it to the FL runtime.
"""
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
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


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.ModuleDict:
    g_kwargs = config.get("generator_kwargs", {})
    d_kwargs = config.get("discriminator_kwargs", {})
    netG = Generator(**g_kwargs)
    netD = Discriminator(**d_kwargs)
    netG.apply(weights_init)
    netD.apply(weights_init)
    return nn.ModuleDict({"netG": netG, "netD": netD})


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    dataset_type = config.get("dataset", "fake").lower()
    data_path    = config.get("data_path", ".")
    image_size   = config.get("image_size", 64)

    if dataset_type == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_ds = dset.CIFAR10(root=data_path, download=True, transform=transform)
    elif dataset_type == "mnist":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        full_ds = dset.MNIST(root=data_path, download=True, transform=transform)
    elif dataset_type in ("imagenet", "folder", "lfw"):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_ds = dset.ImageFolder(root=data_path, transform=transform)
    else:
        nc = config.get("generator_kwargs", {}).get("nc", 3)
        full_ds = dset.FakeData(
            image_size=(nc, image_size, image_size),
            transform=transforms.ToTensor(),
        )

    val_ratio = config.get("val_ratio", 0.1)
    n_val   = max(1, int(len(full_ds) * val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
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
    model: nn.ModuleDict,
    batch: tuple | list,
    optimizer: dict,
    config: dict,
) -> torch.Tensor:
    """
    D is updated fully inside this function (backward + step).
    errG is returned WITH grad_fn for the FL runtime to backpropagate through G.

    model     : nn.ModuleDict with keys 'netG' and 'netD'
    optimizer : dict with keys 'optimizerG' and 'optimizerD'

    Returns errG (raw loss tensor, grad_fn attached).
    Do NOT call .backward() or .item() on it before passing to the FL runtime.
    """
    device      = next(iter(model.parameters())).device
    real_images = batch[0].to(device)
    batch_size  = real_images.size(0)

    nz           = config.get("nz", 100)
    real_label_v = config.get("real_label", 1.0)
    fake_label_v = config.get("fake_label", 0.0)

    netG       = model["netG"]
    netD       = model["netD"]
    optimizerD = optimizer["optimizerD"]
    criterion  = nn.BCELoss()

    # ── Discriminator update (local: full backward + step happen here) ──
    optimizerD.zero_grad()

    label = torch.full((batch_size,), real_label_v, dtype=real_images.dtype, device=device)
    errD_real = criterion(netD(real_images), label)
    errD_real.backward()

    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake  = netG(noise)
    label.fill_(fake_label_v)
    errD_fake = criterion(netD(fake.detach()), label)
    errD_fake.backward()
    optimizerD.step()

    # ── Generator loss (returned for FL runtime to backward + step on G) ──
    label.fill_(real_label_v)
    errG = criterion(netD(fake), label)
    return errG
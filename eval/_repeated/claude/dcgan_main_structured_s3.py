from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Model architecture  (preserved exactly; constructor now takes explicit dims
# instead of relying on script-level globals)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if (input.is_cuda or input.is_xpu) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
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
    """
    Thin wrapper that holds both the Generator and Discriminator as named
    sub-modules so the FL runtime can serialize / aggregate them as one unit.
    """
    def __init__(self, netG: Generator, netD: Discriminator):
        super(GANModel, self).__init__()
        self.netG = netG
        self.netD = netD


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return a GANModel (Generator + Discriminator).

    Recognised config keys under 'model_kwargs':
        nz   – latent vector size          (default 100)
        ngf  – generator base filter count  (default 64)
        ndf  – discriminator base filter count (default 64)
        nc   – image channels              (default 3)
        ngpu – number of GPUs              (default 1)
    """
    kwargs = config.get("model_kwargs", {})
    nz   = int(kwargs.get("nz",   100))
    ngf  = int(kwargs.get("ngf",   64))
    ndf  = int(kwargs.get("ndf",   64))
    nc   = int(kwargs.get("nc",     3))
    ngpu = int(kwargs.get("ngpu",   1))

    netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
    netG.apply(weights_init)

    netD = Discriminator(ngpu=ngpu, ndf=ndf, nc=nc)
    netD.apply(weights_init)

    return GANModel(netG, netD)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for 'train' or 'val'.

    Recognised config keys:
        local.batch_size      – batch size (default 16)
        data_path             – root directory for datasets (default ".")
        dataset               – one of cifar10 | mnist | imagenet | folder |
                                lfw | lsun  (default "cifar10")
        image_size            – spatial size fed to Resize (default 64)
        workers               – DataLoader num_workers (default 2)
        val_fraction          – fraction of data held out for val (default 0.1)
        lsun_classes          – comma-separated LSUN class names (default "bedroom")
        allow_synthetic_data  – if True, fall back to torch.randn tensors when
                                the real dataset is unavailable (default False)
        synthetic_samples     – number of synthetic samples (default 1000)
    """
    local_cfg    = config.get("local", {})
    batch_size   = local_cfg.get("batch_size", 16)
    data_path    = config.get("data_path", ".")
    dataset_name = config.get("dataset", "cifar10").lower()
    image_size   = config.get("image_size", 64)
    workers      = config.get("workers", 2)
    val_fraction = config.get("val_fraction", 0.1)

    dataset    = None
    load_error = None

    try:
        if dataset_name == "cifar10":
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = dset.CIFAR10(root=data_path, download=True, transform=transform)

        elif dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            dataset = dset.MNIST(root=data_path, download=True, transform=transform)

        elif dataset_name in ("imagenet", "folder", "lfw"):
            if not os.path.isdir(data_path):
                raise FileNotFoundError(
                    f"ImageFolder dataset requires an existing directory; "
                    f"data_path='{data_path}' not found."
                )
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = dset.ImageFolder(root=data_path, transform=transform)

        elif dataset_name == "lsun":
            if not os.path.isdir(data_path):
                raise FileNotFoundError(
                    f"LSUN dataset requires an existing directory; "
                    f"data_path='{data_path}' not found."
                )
            lsun_classes_cfg = config.get("lsun_classes", "bedroom")
            classes = [c + "_train" for c in lsun_classes_cfg.split(",")]
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = dset.LSUN(root=data_path, classes=classes, transform=transform)

        else:
            raise FileNotFoundError(
                f"Unknown dataset '{dataset_name}'. Supported: cifar10, mnist, "
                f"imagenet, folder, lfw, lsun."
            )

    except Exception as exc:
        load_error = exc

    # ------------------------------------------------------------------
    # Synthetic fallback – ONLY when explicitly allowed
    # ------------------------------------------------------------------
    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset '{dataset_name}' could not be loaded from "
                f"data_path='{data_path}': {load_error}. "
                f"Set config['allow_synthetic_data']=True to use a synthetic "
                f"fallback for smoke-testing (never for real training)."
            ) from load_error

        nc              = int(config.get("model_kwargs", {}).get("nc", 3))
        num_samples     = config.get("synthetic_samples", 1000)
        images          = torch.randn(num_samples, nc, image_size, image_size)
        labels          = torch.randint(0, 10, (num_samples,))
        dataset         = torch.utils.data.TensorDataset(images, labels)

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    total      = len(dataset)
    val_size   = max(1, int(total * val_fraction))
    train_size = total - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_ds if split == "train" else val_ds
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        drop_last=True,
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,           # provided by FL runtime; not called here
    config: dict,
) -> torch.Tensor:
    """
    Single GAN forward pass.  Returns the combined loss WITH grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.

    GAN adaptation for a single-optimizer FL setting
    -------------------------------------------------
    * errD is computed with fake.detach() so the discriminator gradient does
      NOT propagate through the generator.
    * errG is computed on the same fake tensor (no detach) so the generator
      gradient DOES propagate through the generator.
    * Combined loss = errD + errG is returned.  When the FL runtime calls
      loss.backward(), netD receives gradients from errD_real + errD_fake,
      and netG receives gradients from errG — matching the original training
      intent despite using a single optimizer.
    """
    device = next(model.parameters()).device
    nz = int(config.get("model_kwargs", {}).get("nz", 100))

    criterion  = nn.BCELoss()
    real_label = 1.0
    fake_label = 0.0

    real_images = batch[0].to(device)
    batch_size  = real_images.size(0)

    # ------------------------------------------------------------------ #
    # Discriminator loss: log D(x)  +  log(1 - D(G(z)))                  #
    # ------------------------------------------------------------------ #
    label_real    = torch.full(
        (batch_size,), real_label, dtype=real_images.dtype, device=device
    )
    output_real   = model.netD(real_images)
    errD_real     = criterion(output_real, label_real)

    noise         = torch.randn(batch_size, nz, 1, 1, device=device)
    fake          = model.netG(noise)

    label_fake    = torch.full(
        (batch_size,), fake_label, dtype=real_images.dtype, device=device
    )
    # detach: gradient of errD_fake stops at netD; does not flow into netG
    output_fake_d = model.netD(fake.detach())
    errD_fake     = criterion(output_fake_d, label_fake)

    errD = errD_real + errD_fake

    # ------------------------------------------------------------------ #
    # Generator loss: log D(G(z))                                         #
    # ------------------------------------------------------------------ #
    label_for_g   = torch.full(
        (batch_size,), real_label, dtype=real_images.dtype, device=device
    )
    # no detach: gradient of errG flows through netD and then netG
    output_fake_g = model.netD(fake)
    errG          = criterion(output_fake_g, label_for_g)

    # Combined loss returned to FL runtime (backward + step handled there)
    loss = errD + errG
    return loss
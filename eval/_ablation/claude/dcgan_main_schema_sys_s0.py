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
from torch.utils.data import DataLoader, random_split, TensorDataset


# ---------------------------------------------------------------------------
# Weight initialiser (preserved from original)
# ---------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Generator (architecture preserved exactly; global vars replaced with args)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
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


# ---------------------------------------------------------------------------
# Discriminator (architecture preserved exactly; global vars replaced with args)
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, ndf, nc, ngpu):
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


# ---------------------------------------------------------------------------
# Combined GAN wrapper exposed to the FL runtime
# ---------------------------------------------------------------------------

class DCGAN(nn.Module):
    """Wraps Generator + Discriminator so the FL runtime can treat the full
    GAN as a single nn.Module (federated model averaging over both sub-nets)."""

    def __init__(self, nz: int = 100, ngf: int = 64, ndf: int = 64,
                 nc: int = 3, ngpu: int = 1):
        super(DCGAN, self).__init__()
        self.nz = nz
        self.netG = Generator(nz, ngf, nc, ngpu)
        self.netD = Discriminator(ndf, nc, ngpu)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

    def forward(self, real_images: torch.Tensor):
        """Single forward pass used by train_step; returns (fake, D_real, D_fake_det, D_fake)."""
        batch_size = real_images.size(0)
        device = real_images.device

        # D on real data
        d_real = self.netD(real_images)

        # G produces fakes; D evaluated twice: detached (for D loss) and live (for G loss)
        noise = torch.randn(batch_size, self.nz, 1, 1, device=device)
        fake = self.netG(noise)
        d_fake_detached = self.netD(fake.detach())
        d_fake = self.netD(fake)

        return d_real, d_fake_detached, d_fake


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate the DCGAN (Generator + Discriminator) from config."""
    kwargs = config.get("model_kwargs", {})
    model = DCGAN(
        nz=kwargs.get("nz", 100),
        ngf=kwargs.get("ngf", 64),
        ndf=kwargs.get("ndf", 64),
        nc=kwargs.get("nc", 3),
        ngpu=kwargs.get("ngpu", 1),
    )
    # Optionally load pretrained weights
    if kwargs.get("netG_path", ""):
        model.netG.load_state_dict(torch.load(kwargs["netG_path"]))
    if kwargs.get("netD_path", ""):
        model.netD.load_state_dict(torch.load(kwargs["netD_path"]))
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Supported datasets (config['dataset_name']): cifar10, mnist, folder,
    imagenet, lfw, lsun.  Falls back to synthetic data only when
    config['allow_synthetic_data'] is explicitly True; otherwise raises
    FileNotFoundError so clients never silently train on dummy data.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    dataset_name = config.get("dataset_name", "cifar10").lower()
    image_size = config.get("model_kwargs", {}).get("imageSize", 64)
    nc = config.get("model_kwargs", {}).get("nc", 3)

    dataset = None
    load_error = None

    try:
        if dataset_name in ("imagenet", "folder", "lfw"):
            dataset = dset.ImageFolder(
                root=data_path,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            )
        elif dataset_name == "lsun":
            lsun_classes_str = config.get("lsun_classes", "bedroom")
            lsun_classes = [c.strip() + "_train" for c in lsun_classes_str.split(",")]
            dataset = dset.LSUN(
                root=data_path,
                classes=lsun_classes,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            )
        elif dataset_name == "cifar10":
            dataset = dset.CIFAR10(
                root=data_path,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            )
        elif dataset_name == "mnist":
            dataset = dset.MNIST(
                root=data_path,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]),
            )
        else:
            load_error = ValueError(
                f"Unsupported dataset_name='{dataset_name}'. "
                "Supported values: cifar10, mnist, folder, imagenet, lfw, lsun."
            )
    except Exception as exc:
        load_error = exc
        dataset = None

    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            msg = (
                f"Real dataset '{dataset_name}' could not be loaded from '{data_path}'"
                + (f": {load_error}" if load_error else "")
                + ". Set config['allow_synthetic_data'] = True to permit synthetic data."
            )
            raise FileNotFoundError(msg)
        # Synthetic fallback — only reached when allow_synthetic_data is True
        n_samples = config.get("synthetic_n_samples", 1000)
        images = torch.randn(n_samples, nc, image_size, image_size)
        labels = torch.zeros(n_samples, dtype=torch.long)
        dataset = TensorDataset(images, labels)

    # Split into train / val
    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
    )

    chosen_ds = train_ds if split == "train" else val_ds
    return DataLoader(
        chosen_ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=config.get("workers", 2),
        drop_last=True,   # GAN training requires fixed batch size
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """One combined GAN forward pass.

    Computes:
        errD = BCE(D(real), 1) + BCE(D(G(z)).detach(), 0)   — discriminator loss
        errG = BCE(D(G(z)), 1)                               — generator loss
        loss = errD + errG

    The FL runtime is responsible for loss.backward() and optimizer.step().
    Neither is called here.  The combined loss carries valid gradients for
    both netD (via real / fake-detached paths) and netG (via the live fake path).
    """
    device = next(model.parameters()).device
    criterion = nn.BCELoss()

    real_images = batch[0].to(device)
    batch_size = real_images.size(0)

    # Run the shared forward pass
    d_real, d_fake_detached, d_fake = model(real_images)

    # Discriminator loss
    label_real = torch.full((batch_size,), 1.0, dtype=real_images.dtype, device=device)
    label_fake = torch.full((batch_size,), 0.0, dtype=real_images.dtype, device=device)
    errD_real = criterion(d_real, label_real)
    errD_fake = criterion(d_fake_detached, label_fake)
    errD = errD_real + errD_fake

    # Generator loss  (fake labels appear "real" to fool D)
    label_gen = torch.full((batch_size,), 1.0, dtype=real_images.dtype, device=device)
    errG = criterion(d_fake, label_gen)

    # Combined loss — grad graph intact, no backward/step called
    loss = errD + errG
    return loss
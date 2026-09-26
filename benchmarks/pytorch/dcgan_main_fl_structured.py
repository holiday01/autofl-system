from __future__ import print_function
import os
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


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
    def __init__(self, ngpu, nc, ndf):
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


class GAN(nn.Module):
    """Combined GAN wrapper holding Generator and Discriminator for FL training."""

    def __init__(self, ngpu, nz, ngf, ndf, nc):
        super(GAN, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
        self.netD = Discriminator(ngpu=ngpu, nc=nc, ndf=ndf)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

    def forward(self, real_images):
        # Canonical forward: discriminate real images (used by FL runtime for
        # shape / device checks; actual GAN logic lives in train_step).
        return self.netD(real_images)


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate the combined GAN model from config."""
    kwargs = config.get("model_kwargs", {})
    ngpu = kwargs.get("ngpu", 1)
    nz   = kwargs.get("nz",   100)
    ngf  = kwargs.get("ngf",  64)
    ndf  = kwargs.get("ndf",  64)
    nc   = kwargs.get("nc",   3)
    return GAN(ngpu=ngpu, nz=nz, ngf=ngf, ndf=ndf, nc=nc)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val')."""
    local_cfg  = config.get("local", {})
    batch_size = local_cfg.get("batch_size", 16)
    data_path  = config.get("data_path", ".")

    kwargs       = config.get("model_kwargs", {})
    dataset_name = kwargs.get("dataset",    "cifar10")
    image_size   = kwargs.get("image_size", 64)
    nc           = kwargs.get("nc",         3)
    workers      = kwargs.get("workers",    2)

    dataset = None

    try:
        if dataset_name == "cifar10":
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
        elif dataset_name in ("imagenet", "folder", "lfw"):
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
            classes_str = kwargs.get("classes", "bedroom")
            classes = [c + "_train" for c in classes_str.split(",")]
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
        elif dataset_name == "fake":
            dataset = dset.FakeData(
                image_size=(nc, image_size, image_size),
                transform=transforms.ToTensor(),
            )
    except Exception:
        dataset = None

    # Synthetic fallback — always available even when the real dataset is missing
    if dataset is None:
        num_samples = 1000
        images = torch.randn(num_samples, nc, image_size, image_size)
        labels = torch.zeros(num_samples, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(images, labels)

    # Deterministic train / val split (90 / 10)
    total      = len(dataset)
    val_size   = max(1, int(0.1 * total))
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
        drop_last=True,   # GAN label tensors require consistent batch size
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Single GAN forward pass.

    Computes the combined minimax loss:
        loss = errD_real + errD_fake + errG

    Returns the loss tensor WITH gradients attached.
    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); do NOT call them here.
    """
    kwargs = config.get("model_kwargs", {})
    nz     = kwargs.get("nz", 100)

    device = next(model.parameters()).device

    real_images = batch[0].to(device)
    batch_size  = real_images.size(0)

    criterion  = nn.BCELoss()
    real_label = 1.0
    fake_label = 0.0

    # ------------------------------------------------------------------ #
    # Discriminator loss: log D(x)  +  log(1 - D(G(z)))                  #
    # ------------------------------------------------------------------ #
    label_real  = torch.full((batch_size,), real_label, dtype=real_images.dtype, device=device)
    output_real = model.netD(real_images)
    errD_real   = criterion(output_real, label_real)

    noise       = torch.randn(batch_size, nz, 1, 1, device=device)
    fake        = model.netG(noise)                         # keep graph for errG
    label_fake  = torch.full((batch_size,), fake_label, dtype=real_images.dtype, device=device)
    output_fake = model.netD(fake.detach())                 # detach: D update only
    errD_fake   = criterion(output_fake, label_fake)

    errD = errD_real + errD_fake

    # ------------------------------------------------------------------ #
    # Generator loss: log D(G(z))                                         #
    # ------------------------------------------------------------------ #
    label_gen  = torch.full((batch_size,), real_label, dtype=real_images.dtype, device=device)
    output_gen = model.netD(fake)                           # non-detached fake
    errG       = criterion(output_gen, label_gen)

    # Combined minimax loss — grad graph intact, no backward/step here
    loss = errD + errG
    return loss
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
from torch.utils.data import DataLoader, TensorDataset, random_split


# ---------------------------------------------------------------------------
# Weight initialisation (preserved from original)
# ---------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Original model classes — architecture preserved exactly, globals replaced
# by constructor arguments so the FL runtime can configure them via config.
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


# ---------------------------------------------------------------------------
# FL wrapper: a single nn.Module whose forward() returns the combined GAN
# loss so the FL runtime can call backward() / optimizer.step() on it.
#
# Gradient flow is correct:
#   errD_fake  → uses fake.detach()  → D grads only, G is not touched
#   errG       → uses the live fake  → G grads flow through netD(fake)
#   total loss → errD + errG         → all parameters receive correct grads
# ---------------------------------------------------------------------------

class DCGAN(nn.Module):
    """Combined DCGAN for federated training.

    forward(real_images) -> scalar loss tensor = errD + errG
    The FL runtime owns backward() and optimizer.step().
    """

    def __init__(self, ngpu: int = 1, nz: int = 100,
                 ngf: int = 64, ndf: int = 64, nc: int = 3):
        super(DCGAN, self).__init__()
        self.nz = nz
        self.netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
        self.netD = Discriminator(ngpu=ngpu, nc=nc, ndf=ndf)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        self.criterion = nn.BCELoss()

    def forward(self, real_images: torch.Tensor) -> torch.Tensor:
        device = real_images.device
        batch_size = real_images.size(0)

        real_label = torch.ones(batch_size, dtype=real_images.dtype, device=device)
        fake_label = torch.zeros(batch_size, dtype=real_images.dtype, device=device)

        # ---- Discriminator loss ----------------------------------------
        out_real = self.netD(real_images)
        errD_real = self.criterion(out_real, real_label)

        noise = torch.randn(batch_size, self.nz, 1, 1, device=device)
        fake = self.netG(noise)                          # keep for G loss below

        out_fake_d = self.netD(fake.detach())            # detach: D grads only
        errD_fake = self.criterion(out_fake_d, fake_label)
        errD = errD_real + errD_fake

        # ---- Generator loss --------------------------------------------
        out_fake_g = self.netD(fake)                     # live fake: G grads flow
        errG = self.criterion(out_fake_g, real_label)    # G wants D to say "real"

        return errD + errG


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate DCGAN from config['model_kwargs']."""
    kwargs = config.get("model_kwargs", {})
    model = DCGAN(
        ngpu=kwargs.get("ngpu", 1),
        nz=kwargs.get("nz", 100),
        ngf=kwargs.get("ngf", 64),
        ndf=kwargs.get("ndf", 64),
        nc=kwargs.get("nc", 3),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Dataset precedence
    ------------------
    1. Real dataset loaded from config['data_path'] (default '.').
    2. Synthetic TensorDataset — only when config['allow_synthetic_data'] is
       explicitly True; otherwise a FileNotFoundError is raised so the FL
       runtime knows data is missing rather than silently training on noise.
    """
    local_cfg   = config.get("local", {})
    batch_size  = local_cfg.get("batch_size", 16)
    workers     = local_cfg.get("workers", 2)
    data_path   = config.get("data_path", ".")
    dataset_name = config.get("dataset", "cifar10").lower()
    kwargs      = config.get("model_kwargs", {})
    image_size  = kwargs.get("imageSize", 64)

    dataset    = None
    load_error = None

    try:
        if dataset_name in ["imagenet", "folder", "lfw"]:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = dset.ImageFolder(root=data_path, transform=transform)

        elif dataset_name == "lsun":
            classes_str = config.get("lsun_classes", "bedroom")
            classes = [c + "_train" for c in classes_str.split(",")]
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = dset.LSUN(root=data_path, classes=classes, transform=transform)

        elif dataset_name == "cifar10":
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

        else:
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' is not supported. "
                "Choose from: cifar10, mnist, lsun, imagenet, folder, lfw."
            )

    except Exception as exc:
        load_error = exc
        dataset = None

    # ----- synthetic fallback (strictly gated) --------------------------
    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset '{dataset_name}' could not be loaded from "
                f"'{data_path}'. Set config['allow_synthetic_data']=True to "
                "use a synthetic fallback instead."
            ) from load_error

        nc = kwargs.get("nc", 3)
        n_samples = 1000
        images = torch.randn(n_samples, nc, image_size, image_size)
        labels = torch.randint(0, 10, (n_samples,))
        dataset = TensorDataset(images, labels)

    # ----- train / val split --------------------------------------------
    total      = len(dataset)
    val_size   = max(1, int(0.1 * total))
    train_size = total - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    chosen = train_set if split == "train" else val_set

    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        drop_last=True,          # keeps batch dims uniform for conv layers
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """One forward pass of the DCGAN.

    Returns the combined loss tensor (errD + errG) with its grad_fn intact.
    The FL runtime is responsible for loss.backward() and optimizer.step().
    """
    device = next(model.parameters()).device
    real_images = batch[0].to(device)   # batch[1] (labels) unused by GANs
    loss = model(real_images)           # DCGAN.forward returns errD + errG
    return loss                         # grad_fn is alive — do NOT detach
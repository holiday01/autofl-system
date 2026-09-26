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
# Model architecture (preserved exactly; globals replaced by constructor args)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
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
    """Thin wrapper that pairs Generator + Discriminator for FL.

    The FL runtime operates on a single nn.Module.  Both sub-networks live
    here so that their parameters are returned by model.parameters() and are
    therefore covered by a single optimizer and by FedAvg aggregation.
    """

    def __init__(self, ngpu: int, nz: int, ngf: int, ndf: int, nc: int):
        super(GANModel, self).__init__()
        self.nz   = nz
        self.netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
        self.netD = Discriminator(ngpu=ngpu, ndf=ndf, nc=nc)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convenience: discriminate a real/fake image batch.
        return self.netD(x)


# ---------------------------------------------------------------------------
# FL interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the GANModel.

    Recognised model_kwargs keys (all optional):
        ngpu (int, default 1)   – number of GPUs for data-parallel
        nz   (int, default 100) – latent vector size
        ngf  (int, default 64)  – generator feature-map base width
        ndf  (int, default 64)  – discriminator feature-map base width
        nc   (int, default 3)   – number of image channels
    """
    kw   = config.get("model_kwargs", {})
    ngpu = int(kw.get("ngpu", 1))
    nz   = int(kw.get("nz",   100))
    ngf  = int(kw.get("ngf",  64))
    ndf  = int(kw.get("ndf",  64))
    nc   = int(kw.get("nc",   3))
    return GANModel(ngpu=ngpu, nz=nz, ngf=ngf, ndf=ndf, nc=nc)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Config keys consumed:
        local.batch_size       (int,  default 16)
        data_path              (str,  default ".")
        dataset                (str,  default "cifar10")
        model_kwargs.image_size(int,  default 64)
        model_kwargs.nc        (int,  default 3)  – used for synthetic fallback
        workers                (int,  default 2)
        lsun_classes           (str,  default "bedroom") – comma-separated
        allow_synthetic_data   (bool, default False)
        synthetic_n_samples    (int,  default 1000)
    """
    batch_size   = config.get("local", {}).get("batch_size", 16)
    data_path    = config.get("data_path", ".")
    dataset_name = config.get("dataset", "cifar10").lower()
    kw           = config.get("model_kwargs", {})
    image_size   = int(kw.get("image_size", 64))
    workers      = int(config.get("workers", 2))

    _norm3 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    _norm1 = transforms.Normalize((0.5,),           (0.5,))

    real_exc = None
    dataset  = None

    try:
        if dataset_name == "cifar10":
            dataset = dset.CIFAR10(
                root=data_path,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    _norm3,
                ]),
            )

        elif dataset_name == "mnist":
            dataset = dset.MNIST(
                root=data_path,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    _norm1,
                ]),
            )

        elif dataset_name in ("imagenet", "folder", "lfw"):
            if not os.path.isdir(data_path):
                raise FileNotFoundError(
                    f"data_path '{data_path}' does not exist (required for '{dataset_name}')."
                )
            dataset = dset.ImageFolder(
                root=data_path,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    _norm3,
                ]),
            )

        elif dataset_name == "lsun":
            lsun_classes_str = config.get("lsun_classes", "bedroom")
            lsun_classes = [c.strip() + "_train" for c in lsun_classes_str.split(",")]
            if not os.path.isdir(data_path):
                raise FileNotFoundError(
                    f"data_path '{data_path}' does not exist (required for 'lsun')."
                )
            dataset = dset.LSUN(
                root=data_path,
                classes=lsun_classes,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    _norm3,
                ]),
            )

        else:
            raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    except Exception as exc:
        real_exc = exc
        dataset  = None

    # ------------------------------------------------------------------ #
    # Synthetic fallback – ONLY when explicitly permitted by the caller.  #
    # ------------------------------------------------------------------ #
    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            msg = (
                f"Real dataset '{dataset_name}' could not be loaded from "
                f"'{data_path}'"
                + (f": {real_exc}" if real_exc else "")
                + ". Set config['allow_synthetic_data'] = True to use "
                  "synthetic data."
            )
            raise FileNotFoundError(msg) from real_exc

        nc        = int(kw.get("nc", 3))
        n_samples = int(config.get("synthetic_n_samples", 1000))
        images    = torch.randn(n_samples, nc, image_size, image_size)
        labels    = torch.zeros(n_samples, dtype=torch.long)
        dataset   = torch.utils.data.TensorDataset(images, labels)

    # ------------------------------------------------------------------ #
    # Train / val split                                                    #
    # ------------------------------------------------------------------ #
    total      = len(dataset)
    val_size   = max(1, int(0.1 * total))
    train_size = total - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    chosen  = train_ds if split == "train" else val_ds
    shuffle = split == "train"
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=True,          # keeps batch sizes uniform for GAN stability
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """Single forward pass for GAN training.  Returns combined loss WITH grad.

    Loss = errD + errG, where:
        errD = BCE(D(real), 1) + BCE(D(G(z).detach()), 0)
        errG = BCE(D(G(z)),    1)

    Gradients flow to D through errD and to G through errG (fake is NOT
    detached for the generator term so the FL runtime's single backward()
    call reaches both sub-networks via the summed loss).

    NOTE: backward() and optimizer.step() are intentionally omitted;
    the FL runtime is responsible for both.
    """
    device = next(model.parameters()).device

    real_images = batch[0].to(device)
    batch_size  = real_images.size(0)

    netG      = model.netG
    netD      = model.netD
    nz        = model.nz
    criterion = nn.BCELoss()

    real_val  = 1.0
    fake_val  = 0.0

    # ---- D on real -------------------------------------------------------
    real_labels = torch.full(
        (batch_size,), real_val, dtype=real_images.dtype, device=device
    )
    errD_real = criterion(netD(real_images), real_labels)

    # ---- D on fake (detach G output so only D grads computed here) -------
    noise       = torch.randn(batch_size, nz, 1, 1, device=device)
    fake        = netG(noise)                        # keep for G loss below
    fake_labels = torch.full(
        (batch_size,), fake_val, dtype=real_images.dtype, device=device
    )
    errD_fake   = criterion(netD(fake.detach()), fake_labels)

    errD        = errD_real + errD_fake

    # ---- G: fool D (fake NOT detached; grads flow back to netG) ----------
    g_labels    = torch.full(
        (batch_size,), real_val, dtype=real_images.dtype, device=device
    )
    errG        = criterion(netD(fake), g_labels)

    # Combined loss – the FL runtime will call .backward() on this tensor.
    return errD + errG
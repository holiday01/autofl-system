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
# Weights initialisation (preserved from original)
# ---------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Model architecture (preserved from original; globals replaced with ctor args)
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


class DCGAN(nn.Module):
    """
    Thin wrapper that holds both Generator and Discriminator as sub-modules so
    that the FL runtime can aggregate / serialize all parameters in one shot.
    """
    def __init__(self, ngpu: int = 1, nz: int = 100,
                 ngf: int = 64, ndf: int = 64, nc: int = 3):
        super(DCGAN, self).__init__()
        self.nz   = nz
        self.ngpu = ngpu
        self.netG = Generator(ngpu, nz, ngf, nc)
        self.netD = Discriminator(ngpu, nc, ndf)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

    def forward(self, x):
        # Convenience passthrough; actual GAN logic lives in train_step.
        return self.netD(x)


# ---------------------------------------------------------------------------
# FL interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return a DCGAN model.

    Relevant config keys (all under config["model_kwargs"]):
        ngpu  (int, default 1)
        nz    (int, default 100)
        ngf   (int, default 64)
        ndf   (int, default 64)
        nc    (int, default 3)   – 1 for MNIST, 3 for RGB datasets
    """
    kwargs = config.get("model_kwargs", {})
    model = DCGAN(
        ngpu=int(kwargs.get("ngpu", 1)),
        nz=int(kwargs.get("nz",   100)),
        ngf=int(kwargs.get("ngf",  64)),
        ndf=int(kwargs.get("ndf",  64)),
        nc=int(kwargs.get("nc",    3)),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Relevant config keys:
        data_path              (str,  default ".")
        dataset                (str,  default "cifar10") – cifar10 | mnist |
                                       lsun | imagenet | folder | lfw
        image_size             (int,  default 64)
        lsun_classes           (list, default ["bedroom"])
        local.batch_size       (int,  default 16)
        allow_synthetic_data   (bool, default False)
        model_kwargs.nc        (int,  default 3)   – needed for synthetic fallback
    """
    batch_size   = config.get("local",   {}).get("batch_size", 16)
    data_path    = config.get("data_path", ".")
    dataset_name = config.get("dataset",   "cifar10").lower()
    image_size   = int(config.get("image_size", 64))
    nc           = int(config.get("model_kwargs", {}).get("nc", 3))

    dataset = None

    # ---- attempt to build the real dataset --------------------------------
    try:
        if dataset_name == "cifar10":
            dataset = dset.CIFAR10(
                root=data_path, download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            )

        elif dataset_name == "mnist":
            dataset = dset.MNIST(
                root=data_path, download=True,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]),
            )

        elif dataset_name == "lsun":
            lsun_classes = [
                c + "_train"
                for c in config.get("lsun_classes", ["bedroom"])
            ]
            dataset = dset.LSUN(
                root=data_path, classes=lsun_classes,
                transform=transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

    except Exception:
        dataset = None  # will be handled below

    # ---- synthetic fallback (only when explicitly permitted) --------------
    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset '{dataset_name}' could not be loaded from "
                f"'{data_path}'. Set config['allow_synthetic_data'] = True "
                "to enable the synthetic-data fallback."
            )

        class _SyntheticImageDataset(torch.utils.data.Dataset):
            def __init__(self, n: int, channels: int, img_size: int):
                self.images = torch.randn(n, channels, img_size, img_size)
                self.labels = torch.zeros(n, dtype=torch.long)

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]

        dataset = _SyntheticImageDataset(1000, nc, image_size)

    # ---- train / val split ------------------------------------------------
    n_total = len(dataset)
    n_val   = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_set if split == "train" else val_set
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        drop_last=True,   # keeps batch size constant; avoids size-1 BN issues
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,          # provided by the FL runtime; not called here
    config: dict,
) -> torch.Tensor:
    """
    One forward pass of the DCGAN training objective.

    Returns the combined loss  errD + errG  with gradients attached.
    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); neither is called here.

    GAN gradient flow
    -----------------
    errD_real  : gradients → D only
    errD_fake  : gradients → D only  (fake is detached from G)
    errG       : gradients → G (and through D, which is unavoidable with a
                 single combined backward pass)
    """
    device = next(model.parameters()).device
    nz     = int(config.get("model_kwargs", {}).get("nz", 100))

    real   = batch[0].to(device)
    bsz    = real.size(0)

    netD      = model.netD
    netG      = model.netG
    criterion = nn.BCELoss()

    real_label = 1.0
    fake_label = 0.0

    # ------------------------------------------------------------------
    # Discriminator loss
    # ------------------------------------------------------------------
    # (a) real samples
    lbl_real   = torch.full((bsz,), real_label, dtype=real.dtype, device=device)
    out_real   = netD(real)
    errD_real  = criterion(out_real, lbl_real)

    # (b) fake samples – detach so D gradients do NOT flow back into G
    noise_d    = torch.randn(bsz, nz, 1, 1, device=device)
    fake_d     = netG(noise_d).detach()          # no grad path to G
    lbl_fake   = torch.full((bsz,), fake_label, dtype=real.dtype, device=device)
    out_fake_d = netD(fake_d)
    errD_fake  = criterion(out_fake_d, lbl_fake)

    errD = errD_real + errD_fake

    # ------------------------------------------------------------------
    # Generator loss  (fresh noise; fake is NOT detached)
    # ------------------------------------------------------------------
    noise_g    = torch.randn(bsz, nz, 1, 1, device=device)
    fake_g     = netG(noise_g)
    lbl_g      = torch.full((bsz,), real_label, dtype=real.dtype, device=device)
    out_fake_g = netD(fake_g)
    errG       = criterion(out_fake_g, lbl_g)

    # Combined loss returned to the FL runtime (grad is attached)
    loss = errD + errG
    return loss
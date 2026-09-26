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
    """Thin wrapper that houses both Generator and Discriminator for FL federation.

    Keeping both sub-networks in one nn.Module allows the FL runtime to
    aggregate and distribute all parameters through a single model handle.
    """

    def __init__(self, ngpu: int, nz: int, ngf: int, ndf: int, nc: int):
        super(GANModel, self).__init__()
        self.nz   = nz
        self.netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
        self.netD = Discriminator(ngpu=ngpu, ndf=ndf, nc=nc)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

    def forward(self, x):
        # Convenience forward: discriminate a real/fake image batch.
        return self.netD(x)


# ---------------------------------------------------------------------------
# FL interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate GANModel from config['model_kwargs']."""
    kwargs = config.get("model_kwargs", {})
    model = GANModel(
        ngpu=kwargs.get("ngpu", 1),
        nz=kwargs.get("nz",   100),
        ngf=kwargs.get("ngf",  64),
        ndf=kwargs.get("ndf",  64),
        nc=kwargs.get("nc",    3),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for *split* ('train' or 'val').

    Dataset loading priority
    ------------------------
    1. Real dataset at config['data_path'] (always attempted first).
    2. Synthetic TensorDataset — only when config['allow_synthetic_data']
       is explicitly True; otherwise raises FileNotFoundError.

    Supported datasets (config['dataset']):
        cifar10 | mnist | imagenet | folder | lfw | lsun
    """
    batch_size   = config.get("local", {}).get("batch_size", 16)
    data_path    = config.get("data_path", ".")
    dataset_name = config.get("dataset", "cifar10")
    image_size   = config.get("model_kwargs", {}).get("imageSize", 64)
    nc           = config.get("model_kwargs", {}).get("nc", 3)
    workers      = config.get("workers", 2)

    _SUPPORTED = {"cifar10", "mnist", "imagenet", "folder", "lfw", "lsun"}
    if dataset_name not in _SUPPORTED:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {sorted(_SUPPORTED)}."
        )

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

    except Exception as exc:
        load_error = exc

    # ---- Synthetic fallback (must be explicitly opted-in) ----
    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset '{dataset_name}' could not be loaded from '{data_path}'. "
                "Set config['allow_synthetic_data'] = True to use synthetic data instead. "
                f"Cause: {load_error}"
            )
        n_synthetic = config.get("local", {}).get("synthetic_size", 1000)
        images  = torch.randn(n_synthetic, nc, image_size, image_size)
        labels  = torch.zeros(n_synthetic, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(images, labels)

    # ---- Deterministic train / val split ----
    val_fraction = config.get("val_fraction", 0.1)
    n_total      = len(dataset)
    n_val        = max(1, int(n_total * val_fraction))
    n_train      = n_total - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_set if split == "train" else val_set
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """Single GAN forward pass.  Returns combined loss WITH gradients attached.

    The FL runtime is solely responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.

    Combined-loss gradient semantics
    ---------------------------------
    errD_fake  is computed on  fake.detach()  →  gradients reach D only.
    errG       is computed on  fake  (live)   →  gradients reach G only
                                                  (flowing back through D).
    Summing the two losses lets a single loss.backward() correctly deliver
    independent gradient signals to both sub-networks in one pass.
    """
    device     = next(model.parameters()).device
    nz         = config.get("model_kwargs", {}).get("nz", 100)

    real_images = batch[0].to(device)
    batch_size  = real_images.size(0)

    criterion  = nn.BCELoss()
    real_label = 1.0
    fake_label = 0.0

    netD = model.netD
    netG = model.netG

    # ── Discriminator on real images ──────────────────────────────────────
    label_real = torch.full(
        (batch_size,), real_label, dtype=real_images.dtype, device=device
    )
    errD_real = criterion(netD(real_images), label_real)

    # ── Discriminator on fake images (detached — D grads only) ───────────
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake  = netG(noise)                                   # keep live for errG
    label_fake = torch.full(
        (batch_size,), fake_label, dtype=real_images.dtype, device=device
    )
    errD_fake = criterion(netD(fake.detach()), label_fake)

    errD = errD_real + errD_fake

    # ── Generator loss (live fake — G grads only, flowing through D) ─────
    label_for_g = torch.full(
        (batch_size,), real_label, dtype=real_images.dtype, device=device
    )
    errG = criterion(netD(fake), label_for_g)

    # Return combined loss; FL runtime calls .backward() + optimizer.step()
    return errD + errG
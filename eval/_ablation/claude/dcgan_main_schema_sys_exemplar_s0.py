"""
Auto-generated FL client module.
Original script: DCGAN training script (PyTorch examples).

Exposes:
  build_model(config)                    -> nn.Module  (DCGAN wrapper: netG + netD)
  build_dataloader(config, split)        -> DataLoader
  train_step(model, batch, opt, config)  -> loss tensor (with grad_fn)

CONTRACT:
  - train_step performs ONE forward pass and returns the raw combined D+G loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach() or .item()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  The FL runtime owns backward(), step(), and metric extraction.

GAN note: standard GAN training alternates separate D and G backward passes.
  The FL runtime requires a single differentiable scalar loss per step.
  train_step computes errD (real + fake) + errG and returns their sum so the
  FL runtime can call backward() and optimizer.step() once over all parameters.
"""
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
# Model architecture (preserved from original; globals replaced with args)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=3):
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
    def __init__(self, ngpu=1, ndf=64, nc=3):
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
    FL-friendly wrapper that holds both Generator (netG) and Discriminator (netD).
    forward() is provided for completeness but train_step accesses sub-modules
    directly so that D and G losses can be computed in a single differentiable graph.
    """

    def __init__(self, ngpu=1, nz=100, ngf=64, ndf=64, nc=3):
        super(DCGAN, self).__init__()
        self.nz = nz
        self.netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
        self.netD = Discriminator(ngpu=ngpu, ndf=ndf, nc=nc)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

    def forward(self, real_images):
        """Return (D(real), G(z), D(G(z))) for a batch of real images."""
        device = real_images.device
        batch_size = real_images.size(0)
        noise = torch.randn(batch_size, self.nz, 1, 1, device=device)
        fake = self.netG(noise)
        real_out = self.netD(real_images)
        fake_out = self.netD(fake)
        return real_out, fake, fake_out


# ---------------------------------------------------------------------------
# FL Interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the DCGAN (netG + netD) wrapper."""
    kwargs = config.get("model_kwargs", {})
    return DCGAN(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ('train' or 'val').

    Supported config keys
    ---------------------
    dataset        : 'cifar10' | 'mnist' | 'imagenet' | 'folder' | 'lfw' | 'lsun' | 'fake'
                     (default: 'cifar10')
    data_path      : path passed to the torchvision dataset (default: '.')
    image_size     : spatial size fed to Resize/CenterCrop (default: 64)
    nc             : number of image channels for synthetic fallback (default: 3)
    classes        : comma-separated LSUN class names (default: 'bedroom')
    val_ratio      : fraction of data reserved for validation (default: 0.1)
    seed           : random_split seed (default: 42)
    synthetic_n    : number of synthetic samples when fallback is active (default: 1000)
    allow_synthetic_data : must be True to allow synthetic fallback (default: False)
    local.batch_size / batch_size : batch size (default: 64)
    local.num_workers / workers   : DataLoader workers (default: 2)
    local.pin_memory              : pin memory flag (default: True)
    """
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  64))
    num_workers = local.get("num_workers", config.get("workers",      2))
    pin_memory  = local.get("pin_memory",  True)

    data_path    = config.get("data_path", ".")
    dataset_type = config.get("dataset",   "cifar10")
    image_size   = config.get("image_size", 64)
    val_ratio    = config.get("val_ratio",  0.1)
    nc           = config.get("nc",         3)

    transform_rgb = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_gray = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_cifar = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_dataset = None

    if dataset_type in ('imagenet', 'folder', 'lfw'):
        if not os.path.isdir(data_path):
            if not config.get("allow_synthetic_data", False):
                raise FileNotFoundError(
                    f"data_path '{data_path}' does not exist for dataset='{dataset_type}'. "
                    "Set config['allow_synthetic_data']=True to use synthetic data instead."
                )
        else:
            full_dataset = dset.ImageFolder(root=data_path, transform=transform_rgb)

    elif dataset_type == 'lsun':
        classes_str = config.get("classes", "bedroom")
        lsun_classes = [c + '_train' for c in classes_str.split(',')]
        if not os.path.isdir(data_path):
            if not config.get("allow_synthetic_data", False):
                raise FileNotFoundError(
                    f"data_path '{data_path}' does not exist for dataset='lsun'. "
                    "Set config['allow_synthetic_data']=True to use synthetic data instead."
                )
        else:
            full_dataset = dset.LSUN(root=data_path, classes=lsun_classes,
                                     transform=transform_rgb)

    elif dataset_type == 'cifar10':
        try:
            full_dataset = dset.CIFAR10(root=data_path, download=True,
                                        transform=transform_cifar)
        except Exception as exc:
            if not config.get("allow_synthetic_data", False):
                raise FileNotFoundError(
                    f"Failed to load/download CIFAR10 at '{data_path}': {exc}. "
                    "Set config['allow_synthetic_data']=True to use synthetic data instead."
                ) from exc

    elif dataset_type == 'mnist':
        try:
            full_dataset = dset.MNIST(root=data_path, download=True,
                                      transform=transform_gray)
        except Exception as exc:
            if not config.get("allow_synthetic_data", False):
                raise FileNotFoundError(
                    f"Failed to load/download MNIST at '{data_path}': {exc}. "
                    "Set config['allow_synthetic_data']=True to use synthetic data instead."
                ) from exc

    elif dataset_type == 'fake':
        full_dataset = dset.FakeData(
            image_size=(nc, image_size, image_size),
            transform=transforms.ToTensor(),
        )

    # Synthetic fallback — only reached when full_dataset is still None
    if full_dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset unavailable for dataset='{dataset_type}' "
                f"at data_path='{data_path}'. "
                "Set config['allow_synthetic_data']=True to use synthetic data instead."
            )
        n = config.get("synthetic_n", 1000)
        X = torch.randn(n, nc, image_size, image_size)
        y = torch.zeros(n, dtype=torch.long)
        full_dataset = TensorDataset(X, y)

    n_val   = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
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
        drop_last=True,   # keeps batch sizes uniform; important for BN layers in G/D
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass for DCGAN.  Returns combined (errD + errG) WITH grad_fn attached.

    Loss composition
    ----------------
    errD_real = BCE( D(real),  1 )          — D should classify real as real
    errD_fake = BCE( D(G(z)),  0 )          — D should classify fake as fake
    errD      = errD_real + errD_fake
    errG      = BCE( D(G(z)),  1 )          — G wants D to classify fake as real
    loss      = errD + errG                 — single differentiable scalar for FL

    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT .detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    # ---- unpack batch ----
    if isinstance(batch, (list, tuple)):
        real_images = batch[0].to(device)
    elif isinstance(batch, dict):
        real_images = (
            batch.get("image", batch.get("x", batch.get("input")))
        ).to(device)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    batch_size = real_images.size(0)

    # ---- resolve nz from model attribute or config ----
    nz = getattr(model, "nz", config.get("model_kwargs", {}).get("nz", 100))

    netG = model.netG if hasattr(model, "netG") else model
    netD = model.netD if hasattr(model, "netD") else None
    if netD is None:
        raise AttributeError(
            "train_step expects a DCGAN wrapper with .netG and .netD attributes."
        )

    criterion = nn.BCELoss()
    real_label = 1.0
    fake_label = 0.0

    real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
    fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)

    # ---- Discriminator losses ----
    real_out   = netD(real_images)
    errD_real  = criterion(real_out, real_labels)

    noise      = torch.randn(batch_size, nz, 1, 1, device=device)
    fake       = netG(noise)
    # detach fake for D's fake loss so D-gradients don't propagate into G here
    fake_out_D = netD(fake.detach())
    errD_fake  = criterion(fake_out_D, fake_labels)

    errD = errD_real + errD_fake

    # ---- Generator loss ----
    # use the same fake tensor (not detached) so G-gradients flow through
    fake_out_G = netD(fake)
    errG       = criterion(fake_out_G, real_labels)   # G wants D to output 1

    loss = errD + errG
    return loss
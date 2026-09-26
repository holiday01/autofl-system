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
            nn.Tanh()
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


class DCGAN(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, ndf=64, nc=3):
        super(DCGAN, self).__init__()
        self.nz = nz
        self.netG = Generator(ngpu, nz, ngf, nc)
        self.netD = Discriminator(ngpu, ndf, nc)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)


def build_model(config: dict) -> torch.nn.Module:
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
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    dataset_name = config.get("dataset", "cifar10").lower()
    image_size = config.get("model_kwargs", {}).get("image_size", 64)
    nc = config.get("model_kwargs", {}).get("nc", 3)

    dataset = None

    if dataset_name in ["imagenet", "folder", "lfw"]:
        if os.path.isdir(data_path):
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = dset.ImageFolder(root=data_path, transform=transform)

    elif dataset_name == "lsun":
        if os.path.isdir(data_path):
            classes_str = config.get("classes", "bedroom")
            classes = [c + "_train" for c in classes_str.split(",")]
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = dset.LSUN(root=data_path, classes=classes, transform=transform)

    elif dataset_name == "cifar10":
        try:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            dataset = dset.CIFAR10(root=data_path, download=True, transform=transform)
        except Exception:
            pass

    elif dataset_name == "mnist":
        try:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            dataset = dset.MNIST(root=data_path, download=True, transform=transform)
        except Exception:
            pass

    elif dataset_name == "fake":
        dataset = dset.FakeData(
            image_size=(nc, image_size, image_size),
            transform=transforms.ToTensor(),
        )

    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset '{dataset_name}' is unavailable at '{data_path}'. "
                "Set config['allow_synthetic_data'] = True to fall back to synthetic data."
            )

        class _SyntheticDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 1000

            def __getitem__(self, idx):
                return torch.randn(nc, image_size, image_size), torch.tensor(0)

        dataset = _SyntheticDataset()

    n_total = len(dataset)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    chosen = train_set if split == "train" else val_set
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"), num_workers=2)


def train_step(model: torch.nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    nz = model.nz
    criterion = nn.BCELoss()

    real_images = batch[0].to(device)
    batch_size = real_images.size(0)

    # D loss on real images
    label_real = torch.full((batch_size,), 1.0, dtype=real_images.dtype, device=device)
    output_real = model.netD(real_images)
    errD_real = criterion(output_real, label_real)

    # Generate fakes
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake_images = model.netG(noise)

    # D loss on fakes — detach so D gradients don't flow into G via this path
    label_fake = torch.full((batch_size,), 0.0, dtype=real_images.dtype, device=device)
    output_fake = model.netD(fake_images.detach())
    errD_fake = criterion(output_fake, label_fake)

    errD = errD_real + errD_fake

    # G loss — D sees fakes as real
    label_for_G = torch.full((batch_size,), 1.0, dtype=real_images.dtype, device=device)
    output_for_G = model.netD(fake_images)
    errG = criterion(output_for_G, label_for_G)

    return errD + errG
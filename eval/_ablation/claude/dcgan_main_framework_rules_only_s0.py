from __future__ import print_function
import argparse
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

nz = 100
ngf = 64
ndf = 64
nc = 3


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if (input.is_cuda or input.is_xpu) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu):
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


def build_model(config):
    global nz, ngf, ndf, nc
    nz = config.get("nz", 100)
    ngf = config.get("ngf", 64)
    ndf = config.get("ndf", 64)
    nc = config.get("nc", 3)
    ngpu = config.get("ngpu", 1)
    device = torch.device(config.get("device", "cpu"))

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

    return {"netG": netG, "netD": netD}


def build_dataloader(config, split="train"):
    data_path = config.get("data_path", ".")
    batch_size = config.get("local", {}).get("batch_size", 16)
    image_size = config.get("imageSize", 64)
    dataset_type = config.get("dataset", "folder")

    if dataset_type == "cifar10":
        full_dataset = dset.CIFAR10(
            root=data_path,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
    elif dataset_type == "mnist":
        full_dataset = dset.MNIST(
            root=data_path,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
    elif dataset_type == "fake":
        full_dataset = dset.FakeData(
            image_size=(3, image_size, image_size),
            transform=transforms.ToTensor(),
        )
    else:
        full_dataset = dset.ImageFolder(
            root=data_path,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )

    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    chosen = train_dataset if split == "train" else val_dataset
    return torch.utils.data.DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
    )


def train_step(model, batch, optimizer, config):
    device = torch.device(config.get("device", "cpu"))
    nz_val = config.get("nz", 100)

    netG = model["netG"]
    netD = model["netD"]
    optimizerD = optimizer["optimizerD"]
    optimizerG = optimizer["optimizerG"]

    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0

    real_cpu = batch[0].to(device)
    batch_size = real_cpu.size(0)
    label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)

    netD.zero_grad()
    output = netD(real_cpu)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    noise = torch.randn(batch_size, nz_val, 1, 1, device=device)
    fake = netG(noise)
    label.fill_(fake_label)
    output = netD(fake.detach())
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    optimizerD.step()

    netG.zero_grad()
    label.fill_(real_label)
    output = netD(fake)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()

    return {
        "loss_D": errD.item(),
        "loss_G": errG.item(),
        "D_x": D_x,
        "D_G_z1": D_G_z1,
        "D_G_z2": D_G_z2,
    }
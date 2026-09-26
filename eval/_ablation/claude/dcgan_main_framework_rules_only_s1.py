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
    def __init__(self, netG, netD):
        super(GANModel, self).__init__()
        self.netG = netG
        self.netD = netD


def build_model(config):
    nz = config.get("nz", 100)
    ngf = config.get("ngf", 64)
    ndf = config.get("ndf", 64)
    ngpu = config.get("ngpu", 1)
    nc = 1 if config.get("dataset", "fake") == "mnist" else 3

    netG = Generator(ngpu, nz, ngf, nc)
    netG.apply(weights_init)
    netD = Discriminator(ngpu, ndf, nc)
    netD.apply(weights_init)

    return GANModel(netG, netD)


def build_dataloader(config, split="train"):
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    dataset_name = config.get("dataset", "fake")
    image_size = config.get("image_size", 64)

    if dataset_name in ['imagenet', 'folder', 'lfw']:
        dataset = dset.ImageFolder(root=data_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif dataset_name == 'lsun':
        classes = [c + '_train' for c in config.get("classes", "bedroom").split(',')]
        dataset = dset.LSUN(root=data_path, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif dataset_name == 'mnist':
        dataset = dset.MNIST(root=data_path, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(image_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
    else:
        dataset = dset.FakeData(image_size=(3, image_size, image_size),
                                transform=transforms.ToTensor())

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    chosen = train_dataset if split == "train" else val_dataset
    return torch.utils.data.DataLoader(
        chosen, batch_size=batch_size, shuffle=(split == "train"), num_workers=2
    )


def train_step(model, batch, optimizer, config):
    nz = config.get("nz", 100)
    real_label = 1
    fake_label = 0

    device = next(model.parameters()).device
    netG = model.netG
    netD = model.netD
    optimizerD = optimizer["D"]
    optimizerG = optimizer["G"]

    criterion = nn.BCELoss()

    real_cpu = batch[0].to(device)
    batch_size = real_cpu.size(0)

    netD.zero_grad()
    label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)
    output = netD(real_cpu)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    noise = torch.randn(batch_size, nz, 1, 1, device=device)
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
"""
Auto-generated FL client module for a DCGAN Generator.
Original script: pytorch/examples/dcgan/main.py

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.

NOTE FOR GANS:
  This FL client module is designed to train the Generator (netG).
  The Discriminator (netD) is instantiated within `train_step` using parameters
  and state_dict provided via the `config` dictionary. It acts as a fixed
  evaluator during the Generator's training step, or its state could be
  periodically updated by the FL runtime.
"""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


# --- Helper functions and Classes from original script ---
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        # Removed ngpu from constructor and data_parallel logic,
        # as FL clients typically operate on a single device.
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
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        # Removed ngpu from constructor and data_parallel logic.
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
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds the Generator model for the FL client.
    The client is configured to contribute to training the Generator.
    """
    kwargs = config.get("model_kwargs", {})
    
    # Pass num_channels to Generator's constructor if determined by dataloader
    kwargs["nc"] = config.get("num_channels", kwargs.get("nc", 3))

    model = Generator(**kwargs)
    model.apply(weights_init) # Apply init weights as per DCGAN paper

    # Optionally load pre-trained Generator state from config
    if config.get("initial_netG_state_dict"):
        model.load_state_dict(config["initial_netG_state_dict"])
    
    return model


def build_dataloader(config: dict, split: str = "train") -> torch.utils.data.DataLoader:
    """
    Builds the DataLoader for real images.
    The 'split' argument is handled, though for GANs, usually only one dataset is used.
    """
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)
    image_size  = config.get("image_size", 64)
    dataset_name = config.get("dataset", "fake")
    dataroot = config.get("dataroot", ".") # Fallback to current dir if not specified

    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    dataset = None
    nc = 3 # Default channels
    if dataset_name in ['imagenet', 'folder', 'lfw']:
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose(transform_list + [
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        nc = 3
    elif dataset_name == 'lsun':
        classes = [ c + '_train' for c in config.get("classes", "bedroom").split(',')]
        dataset = dset.LSUN(root=dataroot, classes=classes,
                            transform=transforms.Compose(transform_list + [
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc = 3
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10(root=dataroot, download=True,
                               transform=transforms.Compose(transform_list + [
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc = 3
    elif dataset_name == 'mnist':
        dataset = dset.MNIST(root=dataroot, download=True,
                           transform=transforms.Compose(transform_list + [
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc = 1
    elif dataset_name == 'fake':
        dataset = dset.FakeData(image_size=(3, image_size, image_size),
                                transform=transforms.ToTensor())
        nc = 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not dataset:
        raise RuntimeError(f"Dataset '{dataset_name}' could not be loaded from '{dataroot}'")

    # Store num_channels in config for use in build_model or train_step
    config["num_channels"] = nc

    # For GANs, we typically use the full dataset for 'real' samples,
    # and shuffle for training consistency.
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module, # This is the Generator (netG)
    batch: tuple | list, # This is (real_images, labels) from the DataLoader
    optimizer, # This is the Generator's optimizer (optimizerG)
    config: dict,
) -> torch.Tensor:
    """
    Performs ONE forward pass to compute the Generator's loss.
    The Discriminator (netD) is instantiated and loaded from config,
    acting as a fixed evaluator for the Generator during this step.
    """
    device = next(model.parameters()).device # Get current device from model

    # Instantiate the Discriminator model
    discriminator_kwargs = config.get("discriminator_kwargs", {})
    discriminator_kwargs["nc"] = config.get("num_channels", discriminator_kwargs.get("nc", 3))
    netD = Discriminator(**discriminator_kwargs).to(device)

    # Load Discriminator's state_dict if provided (e.g., from server aggregation)
    if config.get("discriminator_state_dict"):
        netD.load_state_dict(config["discriminator_state_dict"])
    else:
        # If no state_dict, initialize with weights_init (local, not aggregated D)
        netD.apply(weights_init)
    netD.eval() # Set Discriminator to eval mode as it's not being optimized in this step

    criterion = nn.BCELoss()
    real_label = config.get("real_label", 1) # Conventionally 1 for real

    # The FL runtime handles optimizer.zero_grad() before calling train_step
    
    # Get batch data (real images are not directly used for G's loss, but for batch_size)
    batch_data = batch[0]
    batch_size = batch_data.size(0)

    # Generate noise for the Generator
    nz = config.get("nz", 100)
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    
    # Generator's forward pass
    fake = model(noise) # `model` is the Generator

    # Compute Generator's loss: D should classify fake images as real (label=real_label)
    label_for_generator = torch.full((batch_size,), real_label,
                                     dtype=batch_data.dtype, device=device)
    output_from_D_on_fakes = netD(fake)
    errG = criterion(output_from_D_on_fakes, label_for_generator)

    # Return raw loss tensor with grad_fn, FL runtime will call .backward()
    return errG
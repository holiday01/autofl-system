"""
Auto-generated FL client module for a DCGAN (based on pytorch/examples/dcgan).
Original script: [path to original script]

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

NOTE: This FL client module is designed for alternating GAN training,
where the orchestrator calls build_model and train_step separately for
the Generator and Discriminator. The 'config' dictionary is used to pass
parameters and the state_dict of the companion model for each step.
This approach adheres to the strict 'train_step' signature but might be
less efficient than a GAN-specific FL framework.
"""
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


# --- Global Settings (from original script's argparse block) ---
# These are typically configured by the FL orchestration system and passed via 'config'
# For example, manual seed is handled in build_model. cudnn.benchmark might be set globally.
cudnn.benchmark = True # Applying this globally as in the original script.


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu # Kept for consistency, but multi-GPU on client is complex for FL
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
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
        # Original script used data_parallel for ngpu > 1, but for FL,
        # clients usually train on a single device, or this is handled by orchestrator.
        # Keeping it simple by directly calling the main sequential module.
        # if (input.is_cuda or input.is_xpu) and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu=1, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu # Kept for consistency
        self.ndf = ndf
        self.nc = nc
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
        # if (input.is_cuda or input.is_xpu) and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    # Manual seed setup for consistency, if provided by config
    seed = config.get("seed")
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    model_type = config.get("model_type", "discriminator")
    kwargs = config.get("model_kwargs", {})
    nc = config.get("num_channels", 3) # Number of channels from dataloader

    if model_type == "generator":
        model = Generator(nc=nc, **kwargs)
    elif model_type == "discriminator":
        model = Discriminator(nc=nc, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'generator' or 'discriminator'.")
    
    # Apply weights_init as done in the original script for fresh models.
    # In FL, initial weights might also come from the server.
    model.apply(weights_init)

    return model


def build_dataloader(config: dict, split: str = "train") -> torch.utils.data.DataLoader:
    # For GANs, the 'split' parameter is less relevant as typically the entire local dataset
    # is used for training. The original script does not define separate train/val splits.
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    dataset_name = config.get("dataset", "fake")
    data_path = config.get("data_path", ".")
    image_size = config.get("image_size", 64)
    classes = config.get("classes", "bedroom") # For LSUN dataset

    dataset = None
    nc = 3 # Default number of channels, will be updated based on dataset

    if dataset_name in ['imagenet', 'folder', 'lfw']:
        dataset = dset.ImageFolder(root=data_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        nc = 3
    elif dataset_name == 'lsun':
        lsun_classes = [c + '_train' for c in classes.split(',')]
        dataset = dset.LSUN(root=data_path, classes=lsun_classes,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc = 3
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        nc = 3
    elif dataset_name == 'mnist':
        dataset = dset.MNIST(root=data_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)), # MNIST is grayscale
                               ]))
        nc = 1
    elif dataset_name == 'fake':
        dataset = dset.FakeData(image_size=(3, image_size, image_size), # Default to 3 channels for fake
                                transform=transforms.ToTensor())
        nc = 3 # FakeData default is 3 channels

    assert dataset is not None, f"Dataset '{dataset_name}' not found or supported."

    # Update config with number of channels, image size, and batch size.
    # These might be needed for building companion models in train_step or for orchestrator.
    config["num_channels"] = nc
    config["image_size"] = image_size
    config["batch_size"] = batch_size # Pass effective batch_size used for noise generation

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # Always shuffle for training in GANs
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer, # This optimizer is for 'model', but not directly used for step/zero_grad
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass for either Discriminator or Generator.
    Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    
    # Extract GAN specific parameters from config
    model_type = config.get("model_type", "discriminator")
    nz = config.get("nz", 100)
    real_label = config.get("real_label", 1)
    fake_label = config.get("fake_label", 0)
    
    # Companion model parameters for instantiation and loading
    companion_model_kwargs = config.get("companion_model_kwargs", {})
    companion_model_state_dict = config.get("companion_model_state_dict")
    num_channels = config.get("num_channels", 3)
    
    # Determine actual batch size from the incoming data batch
    if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], torch.Tensor):
        current_batch_size = batch[0].size(0)
    else:
        # Fallback to config batch size if batch is empty or malformed
        current_batch_size = config.get("batch_size", 64) 

    criterion = nn.BCELoss() # As in the original script

    if model_type == "discriminator":
        # Current model is Discriminator (netD)
        netD = model
        netD.train() # Ensure discriminator is in training mode

        # Create and load Generator (netG) as companion model for inference
        netG = Generator(nc=num_channels, nz=nz, **companion_model_kwargs)
        if companion_model_state_dict:
            netG.load_state_dict(companion_model_state_dict)
        netG.to(device)
        netG.eval() # Generator should be in evaluation mode when training Discriminator

        # Prepare real data (batch[0] contains images, batch[1] contains labels, but labels unused for input)
        real_cpu = batch[0].to(device)
        
        # Train with real images
        label_real = torch.full((current_batch_size,), real_label, dtype=real_cpu.dtype, device=device)
        output_real = netD(real_cpu)
        errD_real = criterion(output_real, label_real)

        # Generate fake images and train with fake
        noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label_fake = torch.full((current_batch_size,), fake_label, dtype=real_cpu.dtype, device=device)
        # Crucial: detach fake from generator's graph when training discriminator
        output_fake = netD(fake.detach()) 
        errD_fake = criterion(output_fake, label_fake)
        
        errD = errD_real + errD_fake
        return errD

    elif model_type == "generator":
        # Current model is Generator (netG)
        netG = model
        netG.train() # Ensure generator is in training mode

        # Create and load Discriminator (netD) as companion model for inference
        netD = Discriminator(nc=num_channels, **companion_model_kwargs)
        if companion_model_state_dict:
            netD.load_state_dict(companion_model_state_dict)
        netD.to(device)
        netD.eval() # Discriminator should be in evaluation mode when training Generator

        # For generator, we generate noise. The 'batch' (real images) is not used as input to G.
        noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        
        # Fake images are considered 'real' for the generator's objective (G tries to fool D)
        label_for_G = torch.full((current_batch_size,), real_label, dtype=fake.dtype, device=device)
        output_from_D_for_G = netD(fake) # No detach here, we want gradients to flow to G
        errG = criterion(output_from_D_for_G, label_for_G)
        return errG

    else:
        raise ValueError(f"Unknown model_type for train_step: {model_type}")
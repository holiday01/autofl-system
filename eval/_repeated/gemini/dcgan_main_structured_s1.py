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
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import math # For random_split ratio

# --- Helper functions and Model Definitions (copied from original) ---

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, nz: int, ngf: int, nc: int): # Removed ngpu, as it's for data_parallel
        super(Generator, self).__init__()
        # self.ngpu = ngpu # Not needed for FL client single device training
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
        # The original script included nn.parallel.data_parallel logic.
        # For a standard FL client, we assume a single device per client
        # or that parallelism is handled by the FL framework.
        # We simplify by directly calling the main sequential module.
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ndf: int, nc: int): # Removed ngpu
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu # Not needed for FL client single device training
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
        # The original script included nn.parallel.data_parallel logic.
        # For a standard FL client, we assume a single device per client
        # or that parallelism is handled by the FL framework.
        # We simplify by directly calling the main sequential module.
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# --- FL Client Module Functions ---

def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model (either Generator or Discriminator).

    Args:
        config: A dictionary containing model configuration, including:
            - "model_kwargs" (dict): Arguments for the model constructor.
                - "model_type" (str): "Generator" or "Discriminator".
                - "nz" (int, optional): Size of latent vector z for Generator.
                - "ngf" (int, optional): Number of generator filters.
                - "ndf" (int, optional): Number of discriminator filters.
                - "nc" (int, optional): Number of image channels (e.g., 3 for RGB, 1 for grayscale).

    Returns:
        torch.nn.Module: The instantiated Generator or Discriminator model.
    """
    model_kwargs = config.get("model_kwargs", {})
    model_type = model_kwargs.get("model_type", "Generator") # Default to Generator if not specified

    # Default values from original script if not provided in config
    nz = model_kwargs.get("nz", 100)
    ngf = model_kwargs.get("ngf", 64)
    ndf = model_kwargs.get("ndf", 64)
    nc = model_kwargs.get("nc", 3) # Number of channels, will be determined by dataset but needed here.

    if model_type == "Generator":
        model = Generator(nz=nz, ngf=ngf, nc=nc)
    elif model_type == "Discriminator":
        model = Discriminator(ndf=ndf, nc=nc)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'Generator' or 'Discriminator'.")

    model.apply(weights_init)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Args:
        config: A dictionary containing data configuration, including:
            - "local" (dict): Local client configuration.
                - "batch_size" (int, optional): Batch size for the DataLoader. Defaults to 16.
            - "data_path" (str, optional): Path to the dataset root directory. Defaults to ".".
            - "image_size" (int, optional): Desired image size (height/width). Defaults to 64.
            - "dataset_type" (str, optional): Type of dataset (e.g., "cifar10", "mnist", "imagenet", "lsun", "folder", "lfw", "fake"). Defaults to "cifar10".
            - "classes" (str, optional): Comma-separated list of classes for LSUN dataset. Defaults to "bedroom".
            - "dataloader_workers" (int, optional): Number of data loading workers. Defaults to 2.
            - "allow_synthetic_data" (bool, optional): If True, falls back to synthetic data if real data is unavailable. Defaults to False.
            - "train_val_split_ratio" (float, optional): Ratio for training data (e.g., 0.9 for 90% train, 10% val). Defaults to 0.9.
            - "manual_seed" (int, optional): Seed for reproducible train/val split.

        split (str): The requested data split, either "train" or "val".

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the specified split.

    Raises:
        FileNotFoundError: If real data is unavailable and `allow_synthetic_data` is False.
        ValueError: If an unsupported dataset type is specified or split is invalid.
    """
    local_config = config.get("local", {})
    batch_size = local_config.get("batch_size", 16)
    data_path = config.get("data_path", ".")
    image_size = config.get("image_size", 64) # Default from original script
    dataset_type = config.get("dataset_type", "cifar10")
    classes = config.get("classes", "bedroom") # For LSUN
    num_workers = config.get("dataloader_workers", 2)
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    
    # Determine number of channels and appropriate transforms based on dataset_type
    nc = 3 # Default for RGB datasets
    if dataset_type == 'mnist':
        nc = 1
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), # For 1-channel grayscale
        ])
    else: # For RGB datasets (cifar10, imagenet, lsun, folder, lfw, fake)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size), # Original script used CenterCrop for ImageFolder/LSUN, not for CIFAR/MNIST initially but Resize will handle. Let's make it consistent.
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # For 3-channel RGB
        ])

    dataset = None
    data_found = False

    if dataset_type == 'fake':
        if allow_synthetic_data:
            dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transform)
            data_found = True
        else:
            raise FileNotFoundError(
                f"Dataset type is 'fake', but allow_synthetic_data is False. Cannot create fake data."
            )
    else:
        try:
            # Check if dataroot is required but not provided
            if dataset_type in ['imagenet', 'folder', 'lfw', 'lsun'] and not data_path:
                 raise ValueError(f"`data_path` parameter is required for dataset \"{dataset_type}\"")

            if dataset_type in ['imagenet', 'folder', 'lfw']:
                dataset = dset.ImageFolder(root=data_path, transform=transform)
            elif dataset_type == 'lsun':
                lsun_classes = [c + '_train' for c in classes.split(',')]
                dataset = dset.LSUN(root=data_path, classes=lsun_classes, transform=transform)
            elif dataset_type == 'cifar10':
                dataset = dset.CIFAR10(root=data_path, download=True, transform=transform)
            elif dataset_type == 'mnist':
                dataset = dset.MNIST(root=data_path, download=True, transform=transform)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
            if dataset is not None and len(dataset) > 0:
                data_found = True
            else:
                # If dataset was instantiated but empty or problematic
                raise RuntimeError(f"Dataset '{dataset_type}' at '{data_path}' is empty or invalid.")

        except Exception as e:
            if allow_synthetic_data:
                print(f"Warning: Real dataset '{dataset_type}' not found at '{data_path}' or failed to load. "
                      f"Falling back to synthetic data due to allow_synthetic_data=True. Error: {e}")
                # Create synthetic data with the expected shape and transform
                dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transform)
                data_found = True
            else:
                raise FileNotFoundError(
                    f"Real dataset '{dataset_type}' not found at '{data_path}' and allow_synthetic_data is False. "
                    f"Please provide valid data or set allow_synthetic_data=True. Original error: {e}"
                )

    if not data_found:
        raise FileNotFoundError(
            f"Could not load any dataset for type '{dataset_type}'. "
            f"Ensure data_path is correct or allow_synthetic_data is True for fallback."
        )

    # Split dataset into train and validation
    train_ratio = config.get("train_val_split_ratio", 0.9)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    # Ensure reproducibility for the split if a seed is provided
    manual_seed = config.get("manual_seed")
    if manual_seed is not None:
        g = torch.Generator().manual_seed(manual_seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if split == "train":
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif split == "val":
        dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        raise ValueError(f"Invalid split: {split}. Expected 'train' or 'val'.")

    return dataloader


def train_step(model: torch.nn.Module, batch, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    Move tensors to the device of the model parameters.

    Args:
        model: The model to train (either Generator or Discriminator).
        batch: A tuple (images, labels) from the DataLoader.
        optimizer: The optimizer for the current model.
        config: A dictionary containing training configuration, including:
            - "real_label" (int, optional): Label for real data. Defaults to 1.
            - "fake_label" (int, optional): Label for fake data. Defaults to 0.
            - "nz" (int, optional): Size of the latent z vector. Defaults to 100.
            - "generator_kwargs" (dict, optional): Keyword arguments to build Generator (needed when training Discriminator).
            - "generator_state_dict" (dict, optional): State dictionary of the Generator (needed when training Discriminator).
            - "discriminator_kwargs" (dict, optional): Keyword arguments to build Discriminator (needed when training Generator).
            - "discriminator_state_dict" (dict, optional): State dictionary of the Discriminator (needed when training Generator).

    Returns:
        torch.Tensor: The loss tensor with gradients attached.
    """
    device = next(model.parameters()).device
    criterion = nn.BCELoss()

    # Get necessary GAN parameters from config
    real_label = config.get("real_label", 1)
    fake_label = config.get("fake_label", 0)
    nz = config.get("nz", 100) # Size of the latent z vector

    images, _ = batch # Data usually comes as (images, labels), but labels not used directly for input
    
    # Move real images to device
    real_cpu = images.to(device)
    batch_size = real_cpu.size(0)

    # Determine if training Generator or Discriminator based on the `model` type
    if isinstance(model, Discriminator):
        # Training Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        
        # We need the Generator to create fake images. It must be provided via config.
        generator_kwargs = config.get("generator_kwargs", {})
        generator_state_dict = config.get("generator_state_dict")
        
        if generator_state_dict is None:
            raise ValueError("Generator state dict missing in config when training Discriminator. "
                             "Please provide 'generator_state_dict' in the config.")

        # Temporarily instantiate Generator, load its state, and set to eval mode
        # to ensure its parameters are not updated by this optimizer
        temp_netG = Generator(
            nz=generator_kwargs.get("nz", nz),
            ngf=generator_kwargs.get("ngf", 64),
            nc=generator_kwargs.get("nc", 3)
        ).to(device)
        temp_netG.load_state_dict(generator_state_dict)
        temp_netG.eval() # Important: set to eval mode so it doesn't accumulate grads unintentionally

        # Train with real data
        label_real_tensor = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)
        output_real = model(real_cpu)
        errD_real = criterion(output_real, label_real_tensor)

        # Train with fake data
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        with torch.no_grad(): # Ensure generator params are not tracked for D's loss calculation
            fake = temp_netG(noise).detach() # Generate fake images using the auxiliary Generator
        label_fake_tensor = torch.full((batch_size,), fake_label, dtype=real_cpu.dtype, device=device)
        output_fake = model(fake) # Discriminator's output on fake images
        errD_fake = criterion(output_fake, label_fake_tensor)

        errD = errD_real + errD_fake
        return errD

    elif isinstance(model, Generator):
        # Training Generator: maximize log(D(G(z)))
        
        # We need the Discriminator to evaluate generated images. It must be provided via config.
        discriminator_kwargs = config.get("discriminator_kwargs", {})
        discriminator_state_dict = config.get("discriminator_state_dict")

        if discriminator_state_dict is None:
            raise ValueError("Discriminator state dict missing in config when training Generator. "
                             "Please provide 'discriminator_state_dict' in the config.")

        # Temporarily instantiate Discriminator, load its state, and set to eval mode
        temp_netD = Discriminator(
            ndf=discriminator_kwargs.get("ndf", 64),
            nc=discriminator_kwargs.get("nc", 3)
        ).to(device)
        temp_netD.load_state_dict(discriminator_state_dict)
        temp_netD.eval() # Important: set to eval mode

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = model(noise) # Generate fake images using the Generator being trained
        
        # Generator wants Discriminator to classify fake images as real (for generator's loss)
        label_for_G = torch.full((batch_size,), real_label, dtype=fake.dtype, device=device) 
        
        # Discriminator's output on fake images; no_grad here is important as D's params are fixed
        with torch.no_grad(): 
            output_D_on_fake = temp_netD(fake) 
        
        errG = criterion(output_D_on_fake, label_for_G)
        
        return errG
    else:
        raise TypeError(f"Unsupported model type for GAN training: {type(model)}. "
                        "Expected Generator or Discriminator.")
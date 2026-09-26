from __future__ import print_function
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


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, nz: int, ngf: int, nc: int, ngpu: int = 1):
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
        # The original script includes data_parallel. For an FL client,
        # it's usually assumed to run on a single device, so ngpu=1 is common.
        if (input.is_cuda or (hasattr(input, 'is_xpu') and input.is_xpu)) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, list(range(self.ngpu)))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc: int, ndf: int, ngpu: int = 1):
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
        # The original script includes data_parallel. For an FL client,
        # it's usually assumed to run on a single device, so ngpu=1 is common.
        if (input.is_cuda or (hasattr(input, 'is_xpu') and input.is_xpu)) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, list(range(self.ngpu)))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the Discriminator model.
    In a federated GAN setup, the Discriminator (netD) is often the model
    trained on client data, while the Generator (netG) might be a global model
    or handled differently.
    """
    model_kwargs = config.get("model_kwargs", {})

    # Extract model parameters with defaults from original script's argparse
    nc = model_kwargs.get("nc", 3)  # Number of channels, depends on dataset
    ndf = model_kwargs.get("ndf", 64)  # Number of discriminator filters
    ngpu = model_kwargs.get("ngpu", 1)  # Assume single GPU/CPU for FL client

    netD = Discriminator(nc=nc, ndf=ndf, ngpu=ngpu)
    netD.apply(weights_init)

    # Load pretrained weights if specified (e.g., for continued training or initialization)
    if model_kwargs.get("netD_path", ''):
        # Using map_location='cpu' is safer if the checkpoint was saved on GPU
        # but the current client might run on CPU. FL runtime often handles device.
        netD.load_state_dict(torch.load(model_kwargs["netD_path"], map_location='cpu'))

    return netD


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    dataset_name = config.get("dataset", "fake")  # Default to 'fake' for robustness if not specified
    image_size = config.get("image_size", 64)  # Default image size
    workers = config.get("num_workers", 2)  # Default number of data loading workers

    # Set random seed for reproducibility of random_split
    manual_seed = config.get("seed", 42)
    torch.manual_seed(manual_seed)
    random.seed(manual_seed)

    dataset = None
    nc = 3  # Default channels, will be updated based on dataset_name

    # Common transform for 3-channel datasets
    transform_3_channel = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Specific transform for 1-channel datasets (like MNIST)
    transform_1_channel = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Determine dataset type and load
    if dataset_name == 'fake':
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                "Synthetic data is not allowed. Please set 'allow_synthetic_data: True' in config "
                "to use synthetic data or provide a valid 'data_path' and 'dataset'."
            )
        # Use nc from model_kwargs if available, otherwise default to 3
        nc_for_fake = config.get("model_kwargs", {}).get("nc", 3)
        dataset = dset.FakeData(image_size=(nc_for_fake, image_size, image_size),
                                transform=transforms.ToTensor())
        nc = nc_for_fake
    elif dataset_name in ['imagenet', 'folder', 'lfw']:
        if not os.path.exists(data_path):
            if config.get("allow_synthetic_data", False):
                nc_for_fake = config.get("model_kwargs", {}).get("nc", 3)
                dataset = dset.FakeData(image_size=(nc_for_fake, image_size, image_size),
                                        transform=transforms.ToTensor())
                print(f"WARNING: Data path '{data_path}' not found for dataset '{dataset_name}'. Using synthetic data.")
            else:
                raise FileNotFoundError(
                    f"Data path '{data_path}' not found for dataset '{dataset_name}'. "
                    "Set 'allow_synthetic_data: True' in config to use synthetic data instead."
                )
        else:
            dataset = dset.ImageFolder(root=data_path, transform=transform_3_channel)
        nc = 3
    elif dataset_name == 'lsun':
        if not os.path.exists(data_path):
            if config.get("allow_synthetic_data", False):
                nc_for_fake = config.get("model_kwargs", {}).get("nc", 3)
                dataset = dset.FakeData(image_size=(nc_for_fake, image_size, image_size),
                                        transform=transforms.ToTensor())
                print(f"WARNING: Data path '{data_path}' not found for dataset '{dataset_name}'. Using synthetic data.")
            else:
                raise FileNotFoundError(
                    f"Data path '{data_path}' not found for dataset '{dataset_name}'. "
                    "Set 'allow_synthetic_data: True' in config to use synthetic data instead."
                )
        else:
            lsun_classes = [c + '_train' for c in config.get("classes", "bedroom").split(',')]
            dataset = dset.LSUN(root=data_path, classes=lsun_classes, transform=transform_3_channel)
        nc = 3
    elif dataset_name == 'cifar10':
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True) # Ensure path exists for download
        try:
            dataset = dset.CIFAR10(root=data_path, download=True, transform=transform_3_channel)
        except Exception as e:
            if config.get("allow_synthetic_data", False):
                nc_for_fake = config.get("model_kwargs", {}).get("nc", 3)
                dataset = dset.FakeData(image_size=(nc_for_fake, image_size, image_size),
                                        transform=transforms.ToTensor())
                print(f"WARNING: Failed to load CIFAR10 from '{data_path}' ({e}). Using synthetic data.")
            else:
                raise FileNotFoundError(
                    f"Failed to load CIFAR10 from '{data_path}'. "
                    "Set 'allow_synthetic_data: True' in config to use synthetic data instead, or ensure download succeeds."
                ) from e
        nc = 3
    elif dataset_name == 'mnist':
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
        try:
            dataset = dset.MNIST(root=data_path, download=True, transform=transform_1_channel)
        except Exception as e:
            if config.get("allow_synthetic_data", False):
                nc_for_fake = config.get("model_kwargs", {}).get("nc", 1) # MNIST is 1 channel
                dataset = dset.FakeData(image_size=(nc_for_fake, image_size, image_size),
                                        transform=transforms.ToTensor())
                print(f"WARNING: Failed to load MNIST from '{data_path}' ({e}). Using synthetic data.")
            else:
                raise FileNotFoundError(
                    f"Failed to load MNIST from '{data_path}'. "
                    "Set 'allow_synthetic_data: True' in config to use synthetic data instead, or ensure download succeeds."
                ) from e
        nc = 1
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Please choose from 'cifar10', 'lsun', 'mnist', 'imagenet', 'folder', 'lfw', 'fake'.")

    if dataset is None:
        raise RuntimeError("Dataset could not be loaded or created, check configuration and data_path.")

    # Apply random_split to create train/val splits
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Use a fixed generator for random_split to ensure reproducibility
    g = torch.Generator().manual_seed(manual_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

    if split == "train":
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(workers))
    elif split == "val":
        dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=int(workers))
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    return dataloader


def train_step(model: torch.nn.Module, batch, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass for the Discriminator model (netD).
    This function computes the loss for the discriminator by evaluating it on
    both real images from the batch and fake images generated by a Generator.
    Returns the total discriminator loss with gradients attached.
    """
    # model is expected to be the Discriminator (netD)
    device = next(model.parameters()).device

    real_images = batch[0].to(device)
    batch_size = real_images.size(0)

    # Instantiate and prepare the Generator (netG) based on config
    # In FL, the Generator's parameters might be provided by the server or
    # updated separately. For this train_step, it's used to produce fake samples.
    generator_kwargs = config.get("generator_kwargs", {})
    nz = generator_kwargs.get("nz", 100)
    ngf = generator_kwargs.get("ngf", 64)
    nc_gen = generator_kwargs.get("nc", 3)  # Channels for Generator input/output
    ngpu_gen = generator_kwargs.get("ngpu", 1)

    netG = Generator(nz=nz, ngf=ngf, nc=nc_gen, ngpu=ngpu_gen).to(device)

    # If a generator state dict is provided (e.g., from server or previous round), load it
    gen_state_dict = config.get("generator_state_dict", {})
    if gen_state_dict:
        netG.load_state_dict(gen_state_dict)
    else:
        # If no state dict, initialize with weights_init (common for fresh start)
        netG.apply(weights_init)

    netG.eval()  # Generator is typically in eval mode when training the Discriminator

    criterion = nn.BCELoss()
    real_label = 1.
    fake_label = 0.

    model.train() # Ensure discriminator is in training mode

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # Train with real images
    label = torch.full((batch_size,), real_label, dtype=torch.float, device=device) # Labels for real data
    output_real = model(real_images)
    errD_real = criterion(output_real, label)
    # NOTE: loss.backward() is handled by the FL runtime

    # Train with fake images
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(noise).detach()  # Generate fake images and detach to prevent G's gradients
    label.fill_(fake_label)  # Labels for fake data
    output_fake = model(fake)
    errD_fake = criterion(output_fake, label)
    # NOTE: loss.backward() is handled by the FL runtime

    # Total discriminator loss
    errD = errD_real + errD_fake

    # Return the combined loss for the Discriminator.
    # The FL runtime will handle `errD.backward()` and `optimizer.step()`.
    return errD
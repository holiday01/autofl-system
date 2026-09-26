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


# Original weights_init function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# Original Generator class, modified to accept nz, ngf, nc directly
class Generator(nn.Module):
    def __init__(self, ngpu: int, nz: int, ngf: int, nc: int):
        super(Generator, self).__init__()
        self.ngpu = ngpu # Note: ngpu here is for internal data_parallel checks in original, but FL handles parallelism.
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
        # The original script had data_parallel logic, but for FL,
        # the framework typically handles moving the model to the correct device
        # and potentially wrapping it in DataParallel if needed.
        # So we can simplify this.
        return self.main(input)


# Original Discriminator class, modified to accept ndf, nc directly
class Discriminator(nn.Module):
    def __init__(self, ngpu: int, ndf: int, nc: int):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu # Note: ngpu here is for internal data_parallel checks in original, but FL handles parallelism.
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
        # Similar to Generator, simplify data_parallel logic for FL context
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# Wrapper for both Generator and Discriminator to be returned by build_model
class GANModel(nn.Module):
    def __init__(self, nz: int, ngf: int, ndf: int, nc: int, ngpu: int = 1):
        super().__init__()
        self.generator = Generator(ngpu, nz, ngf, nc)
        self.discriminator = Discriminator(ngpu, ndf, nc)

        # Apply weights initialization to both sub-models
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    # A forward method is required for nn.Module, but it won't be directly used
    # by train_step in its typical "single model forward pass" sense for GANs.
    # The actual forward passes for G and D happen within train_step.
    def forward(self, x):
        # This forward is a placeholder as actual GAN logic is in train_step
        # If the FL runtime calls model.forward(), it might be for simple inference
        # or model inspection, not for the complex training loop.
        # Returning the sub-models allows for easier inspection if needed.
        return self.generator, self.discriminator


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the GAN model wrapper (Generator and Discriminator).
    """
    model_kwargs = config.get("model_kwargs", {})
    
    # Default values from the original script if not in config
    nz = model_kwargs.get("nz", 100)
    ngf = model_kwargs.get("ngf", 64)
    ndf = model_kwargs.get("ndf", 64)
    ngpu = model_kwargs.get("ngpu", 1) # FL usually manages device, so ngpu might be conceptual

    # Determine number of channels (nc) based on the dataset type
    dataset_name = config.get("dataset", "cifar10").lower()
    nc = 3 # Default for most datasets
    if dataset_name == 'mnist':
        nc = 1
    # For other datasets like imagenet, lsun, cifar10, fake, nc remains 3
    
    return GANModel(nz=nz, ngf=ngf, ndf=ndf, nc=nc, ngpu=ngpu)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    Includes synthetic data fallback.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    image_size = config.get("image_size", 64)
    dataset_name = config.get("dataset", "cifar10").lower()
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    workers = config.get("local", {}).get("num_workers", 2)
    val_ratio = config.get("val_ratio", 0.1) # Default 10% for validation

    transform_list = [
        transforms.Resize(image_size),
    ]

    nc = 3 # Default channels, updated based on dataset

    if dataset_name == 'mnist':
        nc = 1
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else: # cifar10, imagenet, lsun, folder, lfw, fake
        nc = 3
        # For datasets like ImageFolder or LSUN, CenterCrop is usually applied after Resize
        if dataset_name in ['imagenet', 'folder', 'lfw', 'lsun']:
            transform_list.append(transforms.CenterCrop(image_size))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    transform = transforms.Compose(transform_list)

    dataset = None
    
    # Path for specific datasets (e.g., ImageFolder, LSUN)
    dataset_specific_path = os.path.join(data_path, dataset_name) if dataset_name not in ['cifar10', 'mnist', 'fake'] else data_path

    # Check if a specific dataset directory (like for ImageFolder, LSUN) exists
    # For CIFAR10/MNIST, the base data_path is used for download root
    if dataset_name in ['imagenet', 'folder', 'lfw', 'lsun'] and not os.path.exists(dataset_specific_path):
        if allow_synthetic_data:
            print(f"Warning: Dataset path '{dataset_specific_path}' not found for '{dataset_name}'. Using synthetic data.")
            dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transforms.ToTensor())
        else:
            raise FileNotFoundError(f"Dataset path '{dataset_specific_path}' not found for '{dataset_name}'. Set 'allow_synthetic_data: true' in config to use synthetic data.")

    if dataset is None: # Only proceed if synthetic data wasn't already chosen due to missing path
        if dataset_name == 'imagenet' or dataset_name == 'folder' or dataset_name == 'lfw':
            dataset = dset.ImageFolder(root=dataset_specific_path, transform=transform)
        elif dataset_name == 'lsun':
            lsun_classes = config.get("lsun_classes", "bedroom").split(',')
            classes = [c + '_train' for c in lsun_classes] # Original script uses _train suffix
            dataset = dset.LSUN(root=dataset_specific_path, classes=classes, transform=transform)
        elif dataset_name == 'cifar10':
            os.makedirs(data_path, exist_ok=True) # Ensure directory exists for download
            try:
                dataset = dset.CIFAR10(root=data_path, download=True, transform=transform)
            except Exception as e:
                if allow_synthetic_data:
                    print(f"Warning: CIFAR10 download or load failed ({e}). Using synthetic data.")
                    dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transforms.ToTensor())
                else:
                    raise Exception(f"Failed to load CIFAR10 dataset from '{data_path}'. Error: {e}. Set 'allow_synthetic_data: true' in config to use synthetic data.")
        elif dataset_name == 'mnist':
            os.makedirs(data_path, exist_ok=True) # Ensure directory exists for download
            try:
                dataset = dset.MNIST(root=data_path, download=True, transform=transform)
            except Exception as e:
                if allow_synthetic_data:
                    print(f"Warning: MNIST download or load failed ({e}). Using synthetic data.")
                    dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transforms.ToTensor())
                else:
                    raise Exception(f"Failed to load MNIST dataset from '{data_path}'. Error: {e}. Set 'allow_synthetic_data: true' in config to use synthetic data.")
        elif dataset_name == 'fake':
            dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transforms.ToTensor())
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    if dataset is None: # Final check if dataset couldn't be loaded and wasn't replaced by synthetic
        if allow_synthetic_data:
            print(f"Warning: Dataset '{dataset_name}' could not be loaded, falling back to synthetic data.")
            dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transforms.ToTensor())
        else:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}' and synthetic data is not allowed. Check data_path and dataset configuration.")
    
    # Split dataset into train and validation
    total_len = len(dataset)
    if total_len == 0: # If the dataset is empty, create synthetic data
        if allow_synthetic_data:
            print(f"Warning: Loaded dataset '{dataset_name}' is empty. Using synthetic data to ensure at least one sample.")
            dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transforms.ToTensor(), size=batch_size * 10)
            total_len = len(dataset)
        else:
            raise RuntimeError(f"Loaded dataset '{dataset_name}' is empty and synthetic data is not allowed.")
            
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len
    
    # Ensure there's at least one sample in train_dataset if possible
    if train_len == 0 and total_len > 0:
        train_len = total_len
        val_len = 0
    elif total_len == 0: # This case should be handled by the check above, but for safety
        if allow_synthetic_data:
            print(f"Warning: Dataset too small for split ({total_len} samples). Using synthetic data to ensure non-empty splits.")
            dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transforms.ToTensor(), size=batch_size * 10)
            total_len = len(dataset)
            val_len = int(total_len * val_ratio)
            train_len = total_len - val_len
            if train_len == 0: # If synthetic also turns out tiny
                train_len = 1
                val_len = 0
        else:
            raise RuntimeError(f"Dataset '{dataset_name}' is too small to create splits ({total_len} samples) and synthetic data is not allowed.")

    if train_len + val_len != total_len: # Adjust for any rounding issues
        train_len = total_len - val_len

    manual_seed = config.get("manual_seed", None)
    if manual_seed is not None:
        generator = torch.Generator().manual_seed(manual_seed)
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)
    else:
        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    if split == "train":
        if len(train_dataset) == 0:
            if allow_synthetic_data:
                print(f"Warning: Train split is empty. Using synthetic data for training.")
                train_dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transforms.ToTensor(), size=batch_size * 10)
            else:
                raise RuntimeError("Train split is empty and synthetic data is not allowed.")
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    elif split == "val":
        if len(val_dataset) == 0:
            if allow_synthetic_data:
                print(f"Warning: Validation split is empty. Using synthetic data for validation.")
                val_dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transforms.ToTensor(), size=batch_size * 2)
            else:
                # An empty DataLoader for validation is acceptable if no validation data is available.
                pass 
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")


def train_step(model: GANModel, batch, optimizer: optim.Optimizer, config: dict) -> torch.Tensor:
    """
    Run one training step for the GAN (both Discriminator and Generator).
    Returns the combined loss for D and G.
    """
    device = next(model.parameters()).device # Get model's device

    # Extract components from the GANModel wrapper
    netD = model.discriminator
    netG = model.generator

    # Retrieve parameters from config
    real_label = config.get("real_label", 1.0)
    fake_label = config.get("fake_label", 0.0)
    nz = config.get("model_kwargs", {}).get("nz", 100) # latent vector size
    
    criterion = nn.BCELoss()

    # The FL runtime is expected to call optimizer.zero_grad() *before* train_step.
    # Therefore, no explicit zero_grad() calls are needed here for netD or netG.
    # The gradients will accumulate from all computations on `model.parameters()`.

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # Train with real
    real_cpu = batch[0].to(device)
    batch_size = real_cpu.size(0)
    label = torch.full((batch_size,), real_label,
                       dtype=real_cpu.dtype, device=device)

    output_D_real = netD(real_cpu)
    errD_real = criterion(output_D_real, label)
    # No .backward() here as FL runtime handles it for the returned loss

    # Train with fake
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake_data = netG(noise) # This is a forward pass of G
    label.fill_(fake_label)
    # Detach fake_data for D's update to prevent gradients from flowing to G
    output_D_fake = netD(fake_data.detach()) # This is a forward pass of D
    errD_fake = criterion(output_D_fake, label)
    # No .backward() here

    errD = errD_real + errD_fake
    # No optimizerD.step() here

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    # For G's update, the fake data should NOT be detached, so gradients flow back to G
    label.fill_(real_label)  # fake labels are real for generator cost
    output_G_on_fake = netD(fake_data) # This is another forward pass of D
    errG = criterion(output_G_on_fake, label)
    # No .backward() here
    # No optimizerG.step() here

    # The FL runtime expects a single loss tensor.
    # Combining errD and errG into a single scalar loss.
    # This implies a simultaneous update of G and D based on a combined objective.
    # This is a deviation from standard alternating GAN training, but it is the
    # necessary compromise to fit the specified FL client module interface
    # (single model, single optimizer, single loss return).
    total_loss = errD + errG
    
    return total_loss
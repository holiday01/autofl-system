from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, random_split
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
    def __init__(self, ngpu: int, nz: int, ngf: int, nc: int):
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
        # In a typical FL client setup, ngpu will be 0 or 1, and the FL runtime
        # handles device placement. Data parallelism within a single client
        # is usually not part of the standard FL client module.
        # Keeping the original logic for completeness but it will likely
        # fall into the 'else' branch for single device.
        if self.ngpu > 1 and (input.is_cuda or (hasattr(input, 'is_xpu') and input.is_xpu)):
             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu: int, ndf: int, nc: int):
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
        # Similar to Generator, simplifying for single-device per client.
        if self.ngpu > 1 and (input.is_cuda or (hasattr(input, 'is_xpu') and input.is_xpu)):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    - Uses config.get("model_kwargs", {}) for constructor args.
    """
    model_kwargs = config.get("model_kwargs", {})
    model_type = config.get("model_type", "Discriminator") # Default to Discriminator

    # Extract required parameters from model_kwargs or use defaults
    ngpu = model_kwargs.get("ngpu", 1) # Client typically uses 1 GPU or CPU
    nz = model_kwargs.get("nz", 100)
    ngf = model_kwargs.get("ngf", 64)
    ndf = model_kwargs.get("ndf", 64)
    nc = model_kwargs.get("nc", 3) # Number of channels, important for model architecture

    if model_type == "Generator":
        model = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
    elif model_type == "Discriminator":
        model = Discriminator(ngpu=ngpu, ndf=ndf, nc=nc)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'Generator' or 'Discriminator'.")

    model.apply(weights_init)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    - Read batch_size from config.get("local", {}).get("batch_size", 16).
    - Read data_path from config.get("data_path", ".").
    - Use random_split to produce train/val subsets from a single dataset.
    - Include a synthetic data fallback (torch.randn/randint) for the case
      where the real dataset is unavailable, but it MUST be gated on
      config.get("allow_synthetic_data", False).
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    image_size = config.get("image_size", 64)
    dataset_name = config.get("dataset", "fake")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    val_ratio = config.get("val_ratio", 0.2)
    manual_seed = config.get("manual_seed", random.randint(1, 10000))
    workers = config.get("workers", 2)

    # Set random seed for reproducibility in dataset splitting
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    dataset_instance = None
    nc = 3 # Default channels, will be updated based on dataset_name

    # Define common transformations for real datasets
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    # Specific normalization based on dataset type
    normalize_transform_rgb = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    normalize_transform_grayscale = transforms.Normalize((0.5,), (0.5,))


    if dataset_name in ['imagenet', 'folder', 'lfw']:
        if not os.path.exists(data_path):
            if not allow_synthetic_data:
                raise FileNotFoundError(f"Data path '{data_path}' not found for dataset '{dataset_name}'. "
                                        "Set 'allow_synthetic_data' to True to use synthetic data fallback.")
        if os.path.exists(data_path):
            dataset_instance = dset.ImageFolder(root=data_path,
                                               transform=transforms.Compose(transform_list + [normalize_transform_rgb]))
            nc = 3
        
    elif dataset_name == 'lsun':
        classes = [c + '_train' for c in config.get('classes', 'bedroom').split(',')]
        if not os.path.exists(data_path):
            if not allow_synthetic_data:
                raise FileNotFoundError(f"Data path '{data_path}' not found for dataset '{dataset_name}'. "
                                        "Set 'allow_synthetic_data' to True to use synthetic data fallback.")
        if os.path.exists(data_path):
            dataset_instance = dset.LSUN(root=data_path, classes=classes,
                                        transform=transforms.Compose(transform_list + [normalize_transform_rgb]))
            nc = 3
            
    elif dataset_name == 'cifar10':
        download = not os.path.exists(data_path) # Attempt download if data_path doesn't exist
        
        try:
            # CIFAR10's original transform from script does not include CenterCrop
            cifar_transform_list = [transforms.Resize(image_size), transforms.ToTensor()]
            dataset_instance = dset.CIFAR10(root=data_path, download=download,
                                            transform=transforms.Compose(cifar_transform_list + [normalize_transform_rgb]))
            nc = 3
        except Exception as e:
            if not allow_synthetic_data:
                raise FileNotFoundError(f"Failed to load/download CIFAR10 dataset at '{data_path}' and "
                                        f"'allow_synthetic_data' is False. Error: {e}")
            else:
                print(f"Warning: Failed to load/download CIFAR10. Falling back to synthetic data. Error: {e}")
                dataset_instance = None # Will trigger synthetic fallback
                
    elif dataset_name == 'mnist':
        download = not os.path.exists(data_path) # Attempt download if data_path doesn't exist
        
        try:
            mnist_transform_list = [transforms.Resize(image_size), transforms.ToTensor()]
            dataset_instance = dset.MNIST(root=data_path, download=download,
                                        transform=transforms.Compose(mnist_transform_list + [normalize_transform_grayscale]))
            nc = 1
        except Exception as e:
            if not allow_synthetic_data:
                raise FileNotFoundError(f"Failed to load/download MNIST dataset at '{data_path}' and "
                                        f"'allow_synthetic_data' is False. Error: {e}")
            else:
                print(f"Warning: Failed to load/download MNIST. Falling back to synthetic data. Error: {e}")
                dataset_instance = None # Will trigger synthetic fallback

    # Synthetic data fallback if real data loading failed or 'fake' dataset explicitly requested
    if dataset_instance is None or dataset_name == 'fake':
        if not allow_synthetic_data and dataset_name != 'fake':
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found at '{data_path}' and "
                                    "'allow_synthetic_data' is False.")
        
        if dataset_name != 'fake': # Only print if fallback is triggered, not if 'fake' was requested
            print(f"Using synthetic data fallback for dataset '{dataset_name}'.")

        # Determine number of channels for synthetic data
        if dataset_name == 'mnist':
            nc = 1
        else: # Default for most image datasets or if not specified
            nc = 3
        
        # Replicate dset.FakeData behavior: it generates images of specified size
        # and then applies the given transform.
        if nc == 1:
            synthetic_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize_transform_grayscale
            ])
        else:
            synthetic_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize_transform_rgb
            ])

        dataset_instance = dset.FakeData(
            size=config.get("num_synthetic_samples", 1000),
            image_size=(nc, image_size, image_size),
            transform=synthetic_transform
        )
        print(f"Created torchvision.datasets.FakeData with {len(dataset_instance)} samples.")
        
    # Store 'nc' in config for build_model/train_step if it was not explicitly in model_kwargs
    # This derived NC can be used by train_step for helper model instantiation
    config["_derived_num_channels"] = nc
    
    # Split dataset for train/val
    dataset_size = len(dataset_instance)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    # Ensure consistent splitting
    generator = torch.Generator().manual_seed(manual_seed)
    train_dataset, val_dataset = random_split(dataset_instance, [train_size, val_size], generator=generator)

    if split == "train":
        dataset_to_load = train_dataset
        shuffle_data = True
    elif split == "val":
        dataset_to_load = val_dataset
        shuffle_data = False
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    dataloader = DataLoader(dataset_to_load, batch_size=batch_size, shuffle=shuffle_data, num_workers=workers)
    return dataloader


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    Move tensors to the device of the model parameters.
    """
    # Determine the device of the model parameters
    device = next(model.parameters()).device

    # Get common training parameters from config
    criterion = nn.BCELoss()
    nz = config.get("model_kwargs", {}).get("nz", 100)
    real_label = 1
    fake_label = 0

    # These parameters are for the *other* model (Generator if current model is Discriminator, vice versa)
    # They are needed to instantiate the other model correctly for inference.
    # Prioritize 'nc' from model_kwargs, fallback to derived_num_channels from dataloader
    nc_common = config.get("model_kwargs", {}).get("nc", config.get("_derived_num_channels", 3))
    ngpu_common = config.get("model_kwargs", {}).get("ngpu", 1) # Generally 1 for client for its own model

    model_type = config.get("model_type", "Discriminator")

    if model_type == "Discriminator":
        # Training the Discriminator
        real_cpu = batch[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)

        # Train with real images
        output_real = model(real_cpu)
        errD_real = criterion(output_real, label)

        # Generate fake images using a helper Generator for inference
        gen_model_kwargs = config.get("generator_model_kwargs", {
            "ngpu": ngpu_common,
            "nz": nz,
            "ngf": config.get("model_kwargs", {}).get("ngf", 64), # ngf of the Generator
            "nc": nc_common
        })
        netG = Generator(**gen_model_kwargs).to(device)
        gen_state_dict = config.get("generator_state_dict")
        if gen_state_dict:
            netG.load_state_dict(gen_state_dict)
        netG.eval() # Generator is not being trained in this step, only generating samples

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        with torch.no_grad(): # No gradients needed for Generator parameters
            fake = netG(noise)

        # Train with fake images
        label.fill_(fake_label)
        output_fake = model(fake.detach()) # Detach fake to ensure no gradients flow to Generator
        errD_fake = criterion(output_fake, label)

        errD = errD_real + errD_fake
        return errD

    elif model_type == "Generator":
        # Training the Generator
        # Generator's batch size can be independent of real data, use config's local batch_size
        batch_size = config.get("local", {}).get("batch_size", 16)
        
        # Instantiate a helper Discriminator to evaluate generated samples for inference
        disc_model_kwargs = config.get("discriminator_model_kwargs", {
            "ngpu": ngpu_common,
            "ndf": config.get("model_kwargs", {}).get("ndf", 64), # ndf of the Discriminator
            "nc": nc_common
        })
        netD = Discriminator(**disc_model_kwargs).to(device)
        disc_state_dict = config.get("discriminator_state_dict")
        if disc_state_dict:
            netD.load_state_dict(disc_state_dict)
        netD.eval() # Discriminator is not being trained in this step, only evaluating samples

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = model(noise) # The current model ('model') IS the Generator, generating fake images

        # Generator wants Discriminator to think fakes are real
        label = torch.full((batch_size,), real_label, dtype=fake.dtype, device=device)
        
        with torch.no_grad(): # No gradients needed for Discriminator parameters
            output_fake_eval = netD(fake)
        
        errG = criterion(output_fake_eval, label)
        return errG

    else:
        raise ValueError(f"Unsupported model_type for train_step: {model_type}. Must be 'Generator' or 'Discriminator'.")
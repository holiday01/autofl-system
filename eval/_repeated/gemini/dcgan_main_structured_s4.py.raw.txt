import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torchvision.datasets as dset
import torchvision.transforms as transforms


# Custom weights initialization called on netG and netD (copied from original)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# Generator model (copied from original)
class Generator(nn.Module):
    def __init__(self, ngpu: int = 1, nz: int = 100, ngf: int = 64, nc: int = 3):
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
        if (input.is_cuda or (hasattr(input, 'is_xpu') and input.is_xpu)) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# Discriminator model (copied from original)
class Discriminator(nn.Module):
    def __init__(self, ngpu: int = 1, ndf: int = 64, nc: int = 3):
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
        if (input.is_cuda or (hasattr(input, 'is_xpu') and input.is_xpu)) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def build_model(config: dict) -> torch.nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    model_type = model_kwargs.get("model_type", "Generator") # Default to Generator

    ngpu = model_kwargs.get("ngpu", 1)
    # nc (number of channels) is crucial for both models, often derived from dataset
    # Provide a default, but expect it to be set correctly by FL framework for the dataset
    nc = model_kwargs.get("nc", 3) 

    if model_type == "Generator":
        nz = model_kwargs.get("nz", 100)
        ngf = model_kwargs.get("ngf", 64)
        model = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
    elif model_type == "Discriminator":
        ndf = model_kwargs.get("ndf", 64)
        model = Discriminator(ngpu=ngpu, ndf=ndf, nc=nc)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected 'Generator' or 'Discriminator'.")

    model.apply(weights_init)
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    # Extract parameters from config
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)

    data_kwargs = config.get("data_kwargs", {})
    image_size = data_kwargs.get("image_size", 64)
    dataset_type = data_kwargs.get("dataset_type", "cifar10")
    num_workers = data_kwargs.get("num_workers", 2)
    classes_lsun = data_kwargs.get("classes", "bedroom") # For LSUN dataset
    train_ratio = config.get("local", {}).get("train_ratio", 0.8)
    manual_seed = config.get("seed", 42)

    # Determine number of channels and transforms based on dataset_type
    nc = 3 # Default to 3 channels for most datasets
    if dataset_type == 'mnist':
        nc = 1
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else: # cifar10, lsun, imagenet, folder, lfw, 'fake' (synthetic)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset = None
    real_data_source_attempted = False
    
    # Try to load real dataset if not explicitly 'fake'
    if dataset_type not in ['fake']:
        real_data_source_attempted = True
        try:
            if dataset_type in ['imagenet', 'folder', 'lfw']:
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Data path '{data_path}' not found for dataset type '{dataset_type}'.")
                dataset = dset.ImageFolder(root=data_path, transform=transform)
            elif dataset_type == 'lsun':
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Data path '{data_path}' not found for dataset type '{dataset_type}'.")
                classes = [c + '_train' for c in classes_lsun.split(',')]
                dataset = dset.LSUN(root=data_path, classes=classes, transform=transform)
            elif dataset_type == 'cifar10':
                # download=True for CIFAR10/MNIST
                dataset = dset.CIFAR10(root=data_path, download=True, transform=transform)
            elif dataset_type == 'mnist':
                # download=True for CIFAR10/MNIST
                dataset = dset.MNIST(root=data_path, download=True, transform=transform)
            else:
                raise ValueError(f"Unsupported dataset type for real data: {dataset_type}")

            if dataset is None or len(dataset) == 0:
                raise FileNotFoundError(f"Real dataset '{dataset_type}' loaded empty or failed to initialize.")

        except FileNotFoundError as e:
            print(f"Warning: Real dataset not found or failed to load: {e}")
            dataset = None # Ensure dataset is None to trigger synthetic fallback if needed
        except Exception as e:
            print(f"Warning: An unexpected error occurred while loading real dataset: {e}")
            dataset = None # Ensure dataset is None to trigger synthetic fallback if needed

    # Synthetic data fallback
    if dataset is None: # If real data loading was attempted and failed, or dataset_type was 'fake'
        if allow_synthetic_data:
            print("Falling back to synthetic data generation.")
            num_synthetic_samples = config.get("synthetic_data_samples", 1000)
            # Create synthetic images and dummy labels
            synthetic_images = torch.randn(num_synthetic_samples, nc, image_size, image_size)
            # Dummy labels (0 to 9) for potential compatibility with datasets expecting labels
            synthetic_labels = torch.randint(0, 10, (num_synthetic_samples,)) 
            dataset = TensorDataset(synthetic_images, synthetic_labels)
        else:
            if real_data_source_attempted:
                raise FileNotFoundError(
                    f"Real dataset '{dataset_type}' not available at '{data_path}' "
                    "and 'allow_synthetic_data' is False. Cannot proceed."
                )
            else: # dataset_type was 'fake' and real_data_source_attempted was False, but allow_synthetic_data is False
                raise ValueError(
                    f"Dataset type '{dataset_type}' expects synthetic data, but 'allow_synthetic_data' is False. Cannot proceed."
                )

    # Perform train/val split
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    # Use torch.Generator for reproducible splits
    g = torch.Generator().manual_seed(manual_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

    if split == "train":
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif split == "val":
        dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'val'.")

    return dataloader


def train_step(model: torch.nn.Module, batch: tuple, optimizer: torch.optim.Optimizer, config: dict) -> torch.Tensor:
    # Determine device from model parameters
    device = next(model.parameters()).device

    # Extract training mode and other necessary GAN parameters from config
    local_config = config.get("local", {})
    train_mode = local_config.get("train_mode", "discriminator") # Default to discriminator training

    # Parameters required for generating noise or models
    nz = local_config.get("nz", 100) # Latent vector size
    nc = local_config.get("nc", 3)   # Number of channels for images
    image_size = local_config.get("image_size", 64) # Image size (implicitly determines output size for G)
    real_label = local_config.get("real_label", 1.0)
    fake_label = local_config.get("fake_label", 0.0)
    
    # Batch size might be needed for Generator, as `batch` is not actual images
    batch_size_for_noise = local_config.get("batch_size", 16) 

    # BCELoss is used in the original script
    criterion = nn.BCELoss()

    # --- GAN specific logic ---
    if train_mode == "discriminator":
        netD = model
        # The Generator model's state_dict and kwargs must be provided in the config
        other_model_state_dict = local_config.get("other_model_state_dict")
        other_model_kwargs = local_config.get("other_model_kwargs", {}) # e.g., {'ngpu':1, 'nz':100, 'ngf':64, 'nc':3}

        if other_model_state_dict is None:
            raise ValueError("Discriminator training requires 'other_model_state_dict' (Generator) in config['local'].")

        # Instantiate and load the state of the Generator for generating fake samples
        # This generator is not being updated in this step, so set to eval()
        netG_other = Generator(**other_model_kwargs).to(device)
        netG_other.load_state_dict(other_model_state_dict)
        netG_other.eval() # Ensure no gradients flow back to netG_other

        # train with real data
        real_cpu = batch[0].to(device)
        current_batch_size = real_cpu.size(0)
        label = torch.full((current_batch_size,), real_label, dtype=real_cpu.dtype, device=device)
        
        output_real = netD(real_cpu)
        errD_real = criterion(output_real, label)

        # train with fake data
        noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
        fake = netG_other(noise) # Forward pass of the other model (Generator)
        label.fill_(fake_label)
        output_fake = netD(fake.detach()) # Detach fake to prevent gradients to G
        errD_fake = criterion(output_fake, label)

        errD = errD_real + errD_fake
        return errD

    elif train_mode == "generator":
        netG = model
        # The Discriminator model's state_dict and kwargs must be provided in the config
        other_model_state_dict = local_config.get("other_model_state_dict")
        other_model_kwargs = local_config.get("other_model_kwargs", {}) # e.g., {'ngpu':1, 'ndf':64, 'nc':3}

        if other_model_state_dict is None:
            raise ValueError("Generator training requires 'other_model_state_dict' (Discriminator) in config['local'].")

        # Instantiate and load the state of the Discriminator for evaluating fake samples
        # This discriminator is not being updated in this step, so set to eval()
        netD_other = Discriminator(**other_model_kwargs).to(device)
        netD_other.load_state_dict(other_model_state_dict)
        netD_other.eval() # Ensure no gradients flow back to netD_other

        # For Generator, `batch` typically doesn't contain actual data, we just generate noise
        current_batch_size = batch_size_for_noise 
        # Fake labels are treated as real for the generator's cost function
        label = torch.full((current_batch_size,), real_label, device=device) 

        noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        output = netD_other(fake) # Forward pass of the other model (Discriminator)
        errG = criterion(output, label)
        return errG

    else:
        raise ValueError(f"Unknown train_mode: {train_mode}. Expected 'discriminator' or 'generator'.")
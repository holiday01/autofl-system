"""
Auto-generated FL client module.
Original script: dcgan_train.py (based on pytorch/examples/dcgan)

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

NOTE ON GANS:
  This conversion assumes the FL framework trains one model (Generator or Discriminator)
  at a time. The 'model_type' in the config determines which model is built and
  which training step is performed. For the 'train_step' of one model (e.g., Generator),
  the state_dict of the *other* model (e.g., Discriminator) is expected to be
  provided in the config to enable the loss calculation.
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
import torchvision.utils as vutils # Not used in FL client module directly, but good for context


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
        # In an FL context, device management is typically handled by the runtime,
        # and data_parallel might be managed externally or not used if each client is single-GPU.
        # We simplify this to always use self.main, assuming the model is already on the correct device.
        return self.main(input)


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
        # Same simplification as Generator for data_parallel
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# ── FL Interface ────────────────────────────────────────────────────────

def _get_nc_from_dataset_name(dataset_name: str) -> int:
    """Helper to determine number of channels based on dataset name."""
    dataset_name = str(dataset_name).lower()
    if dataset_name in ['imagenet', 'folder', 'lfw', 'lsun', 'cifar10', 'fake']:
        return 3
    elif dataset_name == 'mnist':
        return 1
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def build_model(config: dict) -> nn.Module:
    model_type = config.get("model_type", "generator").lower()
    kwargs = config.get("model_kwargs", {})

    # Determine nc based on dataset if not explicitly in model_kwargs
    if "nc" not in kwargs and "dataset" in config:
        kwargs["nc"] = _get_nc_from_dataset_name(config["dataset"])
    elif "nc" not in kwargs: # Fallback if dataset not in config either
        kwargs["nc"] = 3 # Default to 3 channels

    if model_type == "generator":
        model = Generator(
            ngpu=kwargs.get("ngpu", 1),
            nz=kwargs.get("nz", 100),
            ngf=kwargs.get("ngf", 64),
            nc=kwargs.get("nc", 3),
        )
        model.apply(weights_init)
        if config.get("netG_path"):
            model.load_state_dict(torch.load(config["netG_path"]))
    elif model_type == "discriminator":
        model = Discriminator(
            ngpu=kwargs.get("ngpu", 1),
            ndf=kwargs.get("ndf", 64),
            nc=kwargs.get("nc", 3),
        )
        model.apply(weights_init)
        if config.get("netD_path"):
            model.load_state_dict(torch.load(config["netD_path"]))
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'generator' or 'discriminator'.")

    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)
    image_size  = config.get("imageSize", 64)
    dataroot    = config.get("dataroot", ".")
    dataset_name = config.get("dataset", "cifar10").lower()
    manual_seed = config.get("seed", 42)

    # Set manual seed for reproducibility
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    # cudnn.benchmark is typically set globally by the FL runtime or user

    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]

    nc = _get_nc_from_dataset_name(dataset_name)
    if nc == 3: # RGB images
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif nc == 1: # Grayscale images
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        raise ValueError(f"Unsupported number of channels: {nc}")

    transform = transforms.Compose(transform_list)

    dataset = None
    if dataset_name in ['imagenet', 'folder', 'lfw']:
        dataset = dset.ImageFolder(root=dataroot, transform=transform)
    elif dataset_name == 'lsun':
        classes = [c + '_train' for c in config.get("classes", "bedroom").split(',')]
        dataset = dset.LSUN(root=dataroot, classes=classes, transform=transform)
    elif dataset_name == 'cifar10':
        dataset = dset.CIFAR10(root=dataroot, download=True, transform=transform)
    elif dataset_name == 'mnist':
        dataset = dset.MNIST(root=dataroot, download=True, transform=transform)
    elif dataset_name == 'fake':
        dataset = dset.FakeData(image_size=(nc, image_size, image_size), transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    if dataset is None:
        raise RuntimeError(f"Failed to load dataset for {dataset_name}")

    # For GANs, there isn't a typical train/val split for the data loader itself.
    # The data loader provides real samples for the discriminator.
    # We'll just return the full dataset for the 'train' split.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"), # Always shuffle for training
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer, # Not directly used for step/zero_grad, but indicates which model's step
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass for either Generator or Discriminator.
    Returns the raw loss tensor WITH grad_fn attached.
    """
    device = next(model.parameters()).device
    model_type = config.get("model_type", "generator").lower()

    # Extract batch data
    # For GANs, batch typically contains (images, labels) but labels are often ignored for real data
    # and generated internally. We only need the real images for D-step.
    # For G-step, we only need batch_size.
    real_images = batch[0].to(device)
    batch_size = real_images.size(0)

    criterion = nn.BCELoss()
    real_label = config.get("real_label", 1)
    fake_label = config.get("fake_label", 0)
    
    # Extract model parameters from config for the *other* model if needed
    nz = config.get("model_kwargs", {}).get("nz", 100)
    ngpu = config.get("model_kwargs", {}).get("ngpu", 1)
    ndf = config.get("model_kwargs", {}).get("ndf", 64)
    ngf = config.get("model_kwargs", {}).get("ngf", 64)
    nc = config.get("model_kwargs", {}).get("nc", 3)


    if model_type == "discriminator":
        # This `model` is the Discriminator (netD)
        # We need the Generator (netG) to create fake samples.
        # Assume netG's state_dict is provided in config.
        generator_state_dict = config.get("generator_state_dict")
        if generator_state_dict is None:
            raise ValueError("Generator state_dict must be provided in config for Discriminator training.")

        netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
        netG.load_state_dict(generator_state_dict)
        netG.eval() # Generator is fixed during D-step

        # 1. Train with real samples
        label_real = torch.full((batch_size,), real_label, dtype=real_images.dtype, device=device)
        output_real = model(real_images)
        errD_real = criterion(output_real, label_real)

        # 2. Train with fake samples
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise).detach() # Detach fake to prevent G from getting gradients from D's update
        label_fake = torch.full((batch_size,), fake_label, dtype=fake.dtype, device=device)
        output_fake = model(fake)
        errD_fake = criterion(output_fake, label_fake)

        # Total Discriminator loss
        errD = errD_real + errD_fake
        return errD

    elif model_type == "generator":
        # This `model` is the Generator (netG)
        # We need the Discriminator (netD) to evaluate fake samples.
        # Assume netD's state_dict is provided in config.
        discriminator_state_dict = config.get("discriminator_state_dict")
        if discriminator_state_dict is None:
            raise ValueError("Discriminator state_dict must be provided in config for Generator training.")

        netD = Discriminator(ngpu=ngpu, ndf=ndf, nc=nc).to(device)
        netD.load_state_dict(discriminator_state_dict)
        netD.eval() # Discriminator is fixed during G-step

        # Generate fake samples
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = model(noise) # model is netG

        # Calculate G's loss based on D's output
        # Generator wants D to classify fakes as real
        label_gen = torch.full((batch_size,), real_label, dtype=fake.dtype, device=device)
        output_gen = netD(fake)
        errG = criterion(output_gen, label_gen)
        return errG

    else:
        raise ValueError(f"Unsupported model_type for train_step: {model_type}")
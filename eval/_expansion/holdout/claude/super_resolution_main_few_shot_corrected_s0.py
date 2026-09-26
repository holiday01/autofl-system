"""
Auto-generated FL client module.
Original script: pytorch/examples super_resolution/main.py

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
from __future__ import print_function
from os import listdir
from os.path import join, exists

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from PIL import Image


# ── Model ────────────────────────────────────────────────────────────────────

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


# ── Dataset ──────────────────────────────────────────────────────────────────

def _is_image_file(filename):
    return any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg"])


def _load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(image_dir, x) for x in listdir(image_dir) if _is_image_file(x)
        ]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        inp = _load_img(self.image_filenames[index])
        target = inp.copy()
        if self.input_transform:
            inp = self.input_transform(inp)
        if self.target_transform:
            target = self.target_transform(target)
        return inp, target

    def __len__(self):
        return len(self.image_filenames)


class _SyntheticSRDataset(data.Dataset):
    """Synthetic fallback: random grayscale tensors for testing without real data."""

    def __init__(self, n=200, crop_size=255, upscale_factor=3):
        self.n = n
        self.lr_size = crop_size // upscale_factor
        self.hr_size = crop_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        inp = torch.randn(1, self.lr_size, self.lr_size)
        target = torch.randn(1, self.hr_size, self.hr_size)
        return inp, target


def _calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def _input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def _target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    upscale_factor = model_kwargs.get("upscale_factor", config.get("upscale_factor", 3))
    return Net(upscale_factor=upscale_factor)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 4))
    pin_memory  = local.get("pin_memory", True)

    upscale_factor = config.get("model_kwargs", {}).get(
        "upscale_factor", config.get("upscale_factor", 3)
    )
    crop_size = _calculate_valid_crop_size(config.get("crop_size", 256), upscale_factor)
    data_path = config.get("data_path", "")

    if data_path and exists(data_path):
        if split == "train":
            image_dir = join(data_path, "train") if exists(join(data_path, "train")) else data_path
            full_dataset = DatasetFromFolder(
                image_dir,
                input_transform=_input_transform(crop_size, upscale_factor),
                target_transform=_target_transform(crop_size),
            )
            val_ratio = config.get("val_ratio", 0.1)
            n_val = max(1, int(len(full_dataset) * val_ratio))
            n_train = len(full_dataset) - n_val
            dataset, _ = random_split(
                full_dataset, [n_train, n_val],
                generator=torch.Generator().manual_seed(config.get("seed", 42)),
            )
        else:
            image_dir = join(data_path, "test") if exists(join(data_path, "test")) else data_path
            dataset = DatasetFromFolder(
                image_dir,
                input_transform=_input_transform(crop_size, upscale_factor),
                target_transform=_target_transform(crop_size),
            )
    else:
        n_synth = config.get("n_synthetic", 200)
        dataset = _SyntheticSRDataset(
            n=n_synth, crop_size=crop_size, upscale_factor=upscale_factor
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    outputs = model(inputs)
    criterion = nn.MSELoss()
    loss = criterion(outputs, targets)
    return loss
from __future__ import print_function
from math import log10

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, TensorDataset, random_split

from os import listdir
from os.path import join, exists
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


# ============================================================================
# Model — preserved exactly from original model.py
# ============================================================================

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


# ============================================================================
# Dataset helpers — preserved exactly from original dataset.py / data.py
# ============================================================================

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data_utils.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)
        ]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.image_filenames)


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def _make_input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def _make_target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


# ============================================================================
# FL interface
# ============================================================================

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the super-resolution Net."""
    model_kwargs = config.get("model_kwargs", {})
    upscale_factor = model_kwargs.get("upscale_factor", 4)
    return Net(upscale_factor=upscale_factor)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for 'train' or 'val'.

    Looks for BSD300 images under:
        <data_path>/BSDS300/images/train/
        <data_path>/BSDS300/images/test/

    Combines both sub-dirs into one pool, then uses random_split (80/20)
    to produce train and val subsets.

    If the dataset is absent and config['allow_synthetic_data'] is True,
    returns a DataLoader backed by random tensors sized to match the model.
    If the dataset is absent and allow_synthetic_data is False (default),
    raises FileNotFoundError.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    upscale_factor = config.get("model_kwargs", {}).get("upscale_factor", 4)
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    lr_size = crop_size // upscale_factor

    bsd_root = join(data_path, "BSDS300", "images")
    train_dir = join(bsd_root, "train")
    test_dir = join(bsd_root, "test")

    def _dir_has_images(d):
        return exists(d) and any(is_image_file(f) for f in listdir(d))

    real_available = _dir_has_images(train_dir) or _dir_has_images(test_dir)

    if not real_available:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"BSD300 dataset not found at '{bsd_root}'. "
                "Expected sub-directories 'train' and/or 'test' containing image files. "
                "Provide the dataset at that path, or set config['allow_synthetic_data'] = True "
                "to use randomly generated tensors instead."
            )
        # --- Synthetic fallback (only reached when explicitly enabled) ---
        n_samples = 128
        inputs = torch.randn(n_samples, 1, lr_size, lr_size)
        targets = torch.randn(n_samples, 1, crop_size, crop_size)
        full_ds = TensorDataset(inputs, targets)
        n_train = max(1, int(0.8 * n_samples))
        n_val = n_samples - n_train
        train_set, val_set = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_set if split == "train" else val_set
        return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))

    # --- Real BSD300 data ---
    in_tf = _make_input_transform(crop_size, upscale_factor)
    tgt_tf = _make_target_transform(crop_size)

    sub_datasets = []
    for d in (train_dir, test_dir):
        if _dir_has_images(d):
            sub_datasets.append(
                DatasetFromFolder(d, input_transform=in_tf, target_transform=tgt_tf)
            )

    full_ds = (
        sub_datasets[0]
        if len(sub_datasets) == 1
        else data_utils.ConcatDataset(sub_datasets)
    )

    n_total = len(full_ds)
    n_train = max(1, int(0.8 * n_total))
    n_val = max(1, n_total - n_train)
    # Re-adjust in edge case where rounding left n_train + n_val != n_total
    n_val = n_total - n_train

    train_set, val_set = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_set if split == "train" else val_set
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(
    model: nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Execute one forward pass and return the loss tensor (grad attached).
    Does NOT call loss.backward() or optimizer.step() — the FL runtime owns those.
    """
    device = next(model.parameters()).device
    inputs, targets = batch[0].to(device), batch[1].to(device)

    criterion = nn.MSELoss()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    return loss  # grad graph intact; backward/step deferred to FL runtime
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn import CrossEntropyLoss

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
import monai.networks.nets


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    spatial_dims = model_kwargs.get("spatial_dims", 3)
    in_channels = model_kwargs.get("in_channels", 1)
    out_channels = model_kwargs.get("out_channels", 2)

    model = monai.networks.nets.DenseNet121(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels
    )
    return model


class RawImageLoaderDataset(Dataset):
    """
    A wrapper around MONAI's ImageDataset to ensure raw image data (NumPy array)
    is returned before any FL client-specific transforms are applied.
    """
    def __init__(self, image_files, labels):
        # We pass transform=None here to get the loaded numpy array before client-specific transforms.
        self._monai_ds = ImageDataset(image_files=image_files, labels=labels, transform=None)
    
    def __len__(self):
        return len(self._monai_ds)
    
    def __getitem__(self, idx):
        # This will return a (NumPy array, label) tuple
        return self._monai_ds[idx]


class RawSyntheticImageDataset(Dataset):
    """
    Generates synthetic 3D image data (NumPy array) and labels.
    """
    def __init__(self, num_samples, input_shape=(96, 96, 96), num_classes=2):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image data (NumPy array) and a label
        dummy_image = np.random.rand(*self.input_shape).astype(np.float32)
        return dummy_image, self.labels[idx]


class TransformedSubsetWrapper(Dataset):
    """
    Wrapper for torch.utils.data.Subset to apply transforms at __getitem__.
    """
    def __init__(self, subset: torch.utils.data.Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # `self.subset[index]` returns (raw_image_np, label)
        raw_image, label = self.subset[index]
        if self.transform:
            transformed_image = self.transform(raw_image)
        else:
            transformed_image = raw_image
        return transformed_image, label

    def __len__(self):
        return len(self.subset)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    num_workers = config.get("local", {}).get("num_workers", 2)
    pin_memory = torch.cuda.is_available()

    # Define transforms
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    # IXI dataset details from the original script
    ixi_data_subdir = os.path.join(data_path, "workspace", "data", "medical", "ixi", "IXI-T1")
    original_image_filenames = [
        "IXI314-IOP-0889-T1.nii.gz", "IXI249-Guys-1072-T1.nii.gz", "IXI609-HH-2600-T1.nii.gz",
        "IXI173-HH-1590-T1.nii.gz", "IXI020-Guys-0700-T1.nii.gz", "IXI342-Guys-0909-T1.nii.gz",
        "IXI134-Guys-0780-T1.nii.gz", "IXI577-HH-2661-T1.nii.gz", "IXI066-Guys-0731-T1.nii.gz",
        "IXI130-HH-1528-T1.nii.gz", "IXI607-Guys-1097-T1.nii.gz", "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz", "IXI344-Guys-0905-T1.nii.gz", "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz", "IXI253-HH-1694-T1.nii.gz", "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz", "IXI585-Guys-1130-T1.nii.gz",
    ]
    original_labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

    image_files = [os.path.join(ixi_data_subdir, f) for f in original_image_filenames]
    dataset_size = len(image_files)

    # Check for existence of real data
    all_files_exist = all(os.path.exists(f) for f in image_files)

    if not all_files_exist:
        if not allow_synthetic_data:
            missing_files = [f for f in image_files if not os.path.exists(f)]
            raise FileNotFoundError(
                f"Real data not found at '{ixi_data_subdir}'. Missing files: {missing_files[:min(5, len(missing_files))]}..."
                f" (total {len(missing_files)} missing out of {dataset_size})."
                " To use synthetic data, set 'allow_synthetic_data': True in the client config."
            )
        else:
            logging.warning("Using synthetic data as real data is not available or incomplete.")
            base_dataset = RawSyntheticImageDataset(dataset_size, input_shape=(96, 96, 96), num_classes=2)
    else:
        logging.info("Using real data from '%s'.", ixi_data_subdir)
        base_dataset = RawImageLoaderDataset(image_files=image_files, labels=original_labels)

    # Apply random_split to create train/val subsets from the base dataset
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size

    seed = config.get("seed", None)
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=g)
    else:
        train_subset, val_subset = random_split(base_dataset, [train_size, val_size])

    if split == "train":
        dataset_for_dataloader = TransformedSubsetWrapper(train_subset, train_transforms)
        shuffle = True
    elif split == "val":
        dataset_for_dataloader = TransformedSubsetWrapper(val_subset, val_transforms)
        shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'val'.")

    dataloader = DataLoader(
        dataset_for_dataloader,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    """
    device = next(model.parameters()).device

    inputs, labels = batch[0].to(device), batch[1].to(device)

    loss_function = CrossEntropyLoss()

    outputs = model(inputs)

    loss = loss_function(outputs, labels)

    return loss
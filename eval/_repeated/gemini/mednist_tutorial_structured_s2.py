import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import monai
from monai.data import ImageDataset
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from monai.networks.nets import DenseNet121


# Helper Dataset to apply transforms after random_split
class TransformWrapperDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # item is (image, label) from the underlying dataset (ImageDataset or SyntheticMRIDataset)
        image, label = self.dataset[index]
        if self.transform:
            # MONAI transforms can handle numpy array or torch.Tensor.
            # ImageDataset by default loads images as numpy arrays before applying its own transform chain.
            # SyntheticMRIDataset also returns numpy arrays for consistency.
            # The MONAI Compose transform will handle this.
            image = self.transform(image)
        return image, label


# Custom Dataset for synthetic data fallback
class SyntheticMRIDataset(Dataset):
    def __init__(self, num_samples, image_dim=(96, 96, 96), num_classes=2):
        self.num_samples = num_samples
        self.image_dim = image_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random 3D image as a float32 numpy array,
        # simulating the output of ImageDataset before MONAI transforms.
        # Scale intensity range to simulate typical image data (0-255 or 0-1)
        image = np.random.rand(*self.image_dim).astype(np.float32) * 255.0
        label = np.random.randint(0, self.num_classes)

        # MONAI ImageDataset typically returns images as numpy arrays and labels as numpy scalar.
        # We match this for consistency with TransformWrapperDataset expectations.
        return image, np.array(label, dtype=np.int64)


def build_model(config: dict) -> torch.nn.Module:
    """
    Instantiate and return the model.
    """
    model_kwargs = config.get("model_kwargs", {})
    # Default parameters based on the original script
    # These can be overridden by model_kwargs
    spatial_dims = model_kwargs.get("spatial_dims", 3)
    in_channels = model_kwargs.get("in_channels", 1)
    out_channels = model_kwargs.get("out_channels", 2)

    model = DenseNet121(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        **{k: v for k, v in model_kwargs.items() if k not in ["spatial_dims", "in_channels", "out_channels"]}
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    allow_synthetic_data = config.get("allow_synthetic_data", False)
    num_workers = config.get("num_workers", 2)
    pin_memory = torch.cuda.is_available()

    # Define transforms as in the original script
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    # Hardcoded image names and labels from the original script's demo data.
    # In a real FL scenario, clients would provide their own data files and labels.
    # Here, we assume the demo data structure is replicated under `data_path`.
    ixi_image_names = [
        "IXI314-IOP-0889-T1.nii.gz", "IXI249-Guys-1072-T1.nii.gz", "IXI609-HH-2600-T1.nii.gz",
        "IXI173-HH-1590-T1.nii.gz", "IXI020-Guys-0700-T1.nii.gz", "IXI342-Guys-0909-T1.nii.gz",
        "IXI134-Guys-0780-T1.nii.gz", "IXI577-HH-2661-T1.nii.gz", "IXI066-Guys-0731-T1.nii.gz",
        "IXI130-HH-1528-T1.nii.gz", "IXI607-Guys-1097-T1.nii.gz", "IXI175-HH-1570-T1.nii.gz",
        "IXI385-HH-2078-T1.nii.gz", "IXI344-Guys-0905-T1.nii.gz", "IXI409-Guys-0960-T1.nii.gz",
        "IXI584-Guys-1129-T1.nii.gz", "IXI253-HH-1694-T1.nii.gz", "IXI092-HH-1436-T1.nii.gz",
        "IXI574-IOP-1156-T1.nii.gz", "IXI585-Guys-1130-T1.nii.gz",
    ]
    ixi_labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

    # Construct the full path to the IXI-T1 dataset subdirectory
    full_ixi_data_path = os.path.join(data_path, "medical", "ixi", "IXI-T1")
    image_files = [os.path.join(full_ixi_data_path, f) for f in ixi_image_names]

    use_synthetic = False
    if not os.path.exists(full_ixi_data_path):
        if allow_synthetic_data:
            logging.info(f"WARNING: Data directory '{full_ixi_data_path}' not found. Using synthetic data.")
            use_synthetic = True
        else:
            raise FileNotFoundError(
                f"Data directory '{full_ixi_data_path}' not found. "
                "Set config['allow_synthetic_data'] to True to use synthetic data instead."
            )
    else:
        # Check if all specified image files exist
        missing_files = [f for f in image_files if not os.path.exists(f)]
        if missing_files:
            if allow_synthetic_data:
                logging.info(f"WARNING: {len(missing_files)}/{len(image_files)} image files not found in '{full_ixi_data_path}'. Using synthetic data.")
                use_synthetic = True
            else:
                raise FileNotFoundError(
                    f"Not all required image files found in '{full_ixi_data_path}'. "
                    f"Missing files: {missing_files[:min(5, len(missing_files))]}... "
                    "Set config['allow_synthetic_data'] to True to use synthetic data instead."
                )

    if use_synthetic:
        # Use custom SyntheticMRIDataset as a fallback
        # It generates images as numpy arrays and labels as numpy scalars,
        # matching what MONAI's ImageDataset would produce before its transforms.
        full_dataset_raw = SyntheticMRIDataset(len(ixi_image_names), image_dim=(96, 96, 96), num_classes=2)
    else:
        # Use MONAI ImageDataset for real data.
        # We pass transform=None here because transforms will be applied by TransformWrapperDataset
        # after the random split, allowing for different train/val transforms.
        full_dataset_raw = ImageDataset(image_files=image_files, labels=ixi_labels, transform=None)

    # Split the full dataset into train and validation subsets using random_split
    total_len = len(full_dataset_raw)
    
    # The original script used a fixed split of 10 train, 10 val from 20 samples.
    # We will use random_split to create a balanced split, e.g., 50/50.
    train_len = int(total_len * 0.5)
    val_len = total_len - train_len
    
    # Use torch.Generator for reproducible random_split if a seed is provided in the config
    generator = None
    seed = config.get("seed")
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    train_subset_raw, val_subset_raw = random_split(full_dataset_raw, [train_len, val_len], generator=generator)

    # Wrap subsets with TransformWrapperDataset to apply different transforms
    if split == "train":
        dataset = TransformWrapperDataset(train_subset_raw, transform=train_transforms)
        shuffle = True  # Original train_loader had shuffle=True
    elif split == "val":
        dataset = TransformWrapperDataset(val_subset_raw, transform=val_transforms)
        shuffle = False # Typically validation data is not shuffled
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'val'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    """
    Run ONE forward pass only. Return the loss tensor WITH grad attached.
    Do NOT call loss.backward() or optimizer.step() — the FL runtime handles that.
    """
    device = next(model.parameters()).device # Get the model's device (e.g., "cuda" or "cpu")
    inputs, labels = batch[0].to(device), batch[1].to(device)

    # Instantiate CrossEntropyLoss as used in the original script
    loss_function = torch.nn.CrossEntropyLoss()

    outputs = model(inputs)
    loss = loss_function(outputs, labels)

    return loss
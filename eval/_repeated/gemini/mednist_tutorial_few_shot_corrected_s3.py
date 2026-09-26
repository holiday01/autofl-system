import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import monai
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
from typing import Callable


class DenseNet121Classifier(nn.Module):
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=2):
        super().__init__()
        self.net = monai.networks.nets.DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        return self.net(x)


# Define transforms once globally
TRAIN_TRANSFORMS = Compose([
    ScaleIntensity(),
    EnsureChannelFirst(),
    Resize((96, 96, 96)),
    RandRotate90(),  # Augmentation for training
])

VAL_TRANSFORMS = Compose([
    ScaleIntensity(),
    EnsureChannelFirst(),
    Resize((96, 96, 96)),
])


class FLMONAIDataSource(Dataset):
    """
    Base Dataset class that either loads real MONAI NIfTI files or generates synthetic data.
    It provides raw data (e.g., a NumPy array for real data, or a dummy tensor for synthetic),
    without applying any transforms. Transforms are applied by a wrapper dataset later.
    """

    def __init__(self, root: str, num_samples: int = 20, num_classes: int = 2, input_raw_shape=(1, 128, 128, 128)):
        self.root = root
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.input_raw_shape = input_raw_shape

        # Original hardcoded image files and labels for IXI dataset
        original_image_filenames = [
            "IXI314-IOP-0889-T1.nii.gz", "IXI249-Guys-1072-T1.nii.gz", "IXI609-HH-2600-T1.nii.gz",
            "IXI173-HH-1590-T1.nii.gz", "IXI020-Guys-0700-T1.nii.gz", "IXI342-Guys-0909-T1.nii.gz",
            "IXI134-Guys-0780-T1.nii.gz", "IXI577-HH-2661-T1.nii.gz", "IXI066-Guys-0731-T1.nii.gz",
            "IXI130-HH-1528-T1.nii.gz", "IXI607-Guys-1097-T1.nii.gz", "IXI175-HH-1570-T1.nii.gz",
            "IXI385-HH-2078-T1.nii.gz", "IXI344-Guys-0905-T1.nii.gz", "IXI409-Guys-0960-T1.nii.gz",
            "IXI584-Guys-1129-T1.nii.gz", "IXI253-HH-1694-T1.nii.gz", "IXI092-HH-1436-T1.nii.gz",
            "IXI574-IOP-1156-T1.nii.gz", "IXI585-Guys-1130-T1.nii.gz",
        ]
        original_labels_array = np.array([
            0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0
        ], dtype=np.int64)

        self.image_files = []
        self.labels = []
        self.is_synthetic = True

        # Try to find real files based on `data_path`
        if self.root and os.path.isdir(self.root):
            found_files = []
            found_labels = []
            for i, fname in enumerate(original_image_filenames):
                full_path = os.path.join(self.root, fname)
                if os.path.exists(full_path):
                    found_files.append(full_path)
                    found_labels.append(original_labels_array[i])

            if len(found_files) > 0:
                self.image_files = found_files
                self.labels = np.array(found_labels, dtype=np.int64)
                self.is_synthetic = False
                print(f"FLMONAIDataSource: Found {len(self.image_files)} real data files in {self.root}.")
            else:
                print(f"FLMONAIDataSource: No original data files found in {self.root}. Using synthetic data.")
        else:
            print(f"FLMONAIDataSource: Data path '{self.root}' not found or not a directory. Using synthetic data.")

        if self.is_synthetic:
            # Generate synthetic data (raw-like tensors)
            self.synthetic_data = []
            for i in range(self.num_samples):
                # Dummy raw tensor that can be processed by MONAI transforms later
                raw_x = torch.randn(*self.input_raw_shape).float()
                y = torch.tensor(i % self.num_classes, dtype=torch.long)
                self.synthetic_data.append((raw_x, y))
        else:
            # For real data, use MONAI's ImageDataset to handle file loading.
            # No transforms are applied here; data will be raw image objects/arrays.
            self.monai_dataset = monai.data.ImageDataset(
                image_files=self.image_files,
                labels=self.labels,
                transform=None  # No transforms applied at this stage
            )

    def __len__(self):
        return len(self.synthetic_data) if self.is_synthetic else len(self.monai_dataset)

    def __getitem__(self, idx):
        if self.is_synthetic:
            return self.synthetic_data[idx]
        else:
            # ImageDataset's __getitem__ will load the NIfTI file and return it
            # as a NumPy array or similar, along with the label.
            return self.monai_dataset[idx]


class TransformedDataset(Dataset):
    """
    A wrapper Dataset that applies a given transform to each item from the underlying dataset.
    This is used to apply split-specific transforms (e.g., train vs. val) after data splitting.
    """

    def __init__(self, dataset: Dataset, transform: Callable = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # The self.dataset here could be a torch.utils.data.Subset,
        # which calls the __getitem__ of its underlying base dataset (FLMONAIDataSource).
        x, y = self.dataset[idx]  # x will be a raw image object/array/tensor, y is a tensor
        if self.transform:
            x = self.transform(x)  # Apply transform (e.g., Compose object)
        return x, y


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Builds and returns the model for the FL client.
    Model parameters can be specified in the config dictionary.
    """
    kwargs = config.get("model_kwargs", {})
    spatial_dims = kwargs.get("spatial_dims", 3)
    in_channels = kwargs.get("in_channels", 1)
    out_channels = kwargs.get("out_channels", 2)
    return DenseNet121Classifier(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Builds and returns a DataLoader for the specified data split ("train" or "val").
    Handles loading real data from `data_path` or generating synthetic data as a fallback.
    Applies appropriate transforms for train/val splits.
    """
    local = config.get("local", {})
    batch_size = local.get("batch_size", config.get("batch_size", 2))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory = local.get("pin_memory", True)
    seed = config.get("seed", 42)
    val_ratio = config.get("val_ratio", 0.5)  # Original script implies ~50/50 split for train/val sets

    data_path = config.get("data_path", "./workspace/data/medical/ixi/IXI-T1")
    dataset_cfg = config.get("dataset_kwargs", {})

    # 1. Instantiate the base data source (provides raw data without transforms)
    full_data_source = FLMONAIDataSource(
        root=data_path,
        num_samples=dataset_cfg.get("num_samples", 20),
        num_classes=dataset_cfg.get("num_classes", 2),
        input_raw_shape=dataset_cfg.get("input_raw_shape", (1, 128, 128, 128))
    )

    n_total = len(full_data_source)
    if n_total == 0:
        raise ValueError("Dataset has 0 samples, cannot create data loader.")

    # Determine lengths for random_split, ensuring non-empty train set if possible
    if n_total == 1:
        n_train = 1
        n_val = 0
        print(f"Dataset has only 1 sample. Assigning to training split. (n_train=1, n_val=0)")
    else:
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        # Ensure n_train is not zero if n_total > 1
        if n_train <= 0:
            n_train = 1
            n_val = n_total - 1
            if n_val < 0:
                n_val = 0
            print(f"Adjusted split to ensure train_size > 0: n_train={n_train}, n_val={n_val}")

    # Ensure lengths sum to n_total (critical for random_split)
    # This also handles cases where n_train or n_val might be 0 after adjustment
    lengths = [n_train, n_val]
    if sum(lengths) != n_total:
        if n_total > 0: # If there's data, ensure it's fully assigned to one split
            if split == "train":
                lengths = [n_total, 0]
            else:
                lengths = [0, n_total]
        else: # n_total is 0
            lengths = [0, 0] # Already handled by earlier check, but for robustness

    # 2. Perform random_split on the base data source's indices
    # This splits the *indices*, giving us subsets of the raw data.
    train_indices, val_indices = random_split(
        range(n_total), lengths,
        generator=torch.Generator().manual_seed(seed),
    )

    # 3. Create Subset objects from the full_data_source using the obtained indices
    # Then wrap these subsets with their specific transforms.
    if split == "train":
        base_ds = Subset(full_data_source, train_indices.indices)
        ds = TransformedDataset(base_ds, transform=TRAIN_TRANSFORMS)
    else:  # split == "val"
        base_ds = Subset(full_data_source, val_indices.indices)
        ds = TransformedDataset(base_ds, transform=VAL_TRANSFORMS)

    # Final check: if the requested split is empty, raise an error
    if len(ds) == 0:
        raise ValueError(f"The '{split}' dataset has 0 samples after splitting. Cannot create DataLoader.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),  # Only shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,  # Not directly used as per contract, but required for signature
    config: dict,
) -> torch.Tensor:
    """
    Performs one forward pass and returns the raw loss tensor with grad_fn attached.
    The FL runtime handles optimizer.zero_grad(), loss.backward(), and optimizer.step().
    """
    device = next(model.parameters()).device

    # Unpack batch (expected to be (inputs, targets))
    if isinstance(batch, (list, tuple)):
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        inputs = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss
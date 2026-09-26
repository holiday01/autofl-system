"""
Auto-generated FL client module.
Original script: mnist_convnet_keras.py

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ── Original source (unchanged) ────────────────────────────────────────
"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
Accelerator: GPU
"""

"""
## Setup
"""

import numpy as np
import keras
from keras import layers

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# ── PyTorch equivalents ─────────────────────────────────────────────────

class MNISTDataset(Dataset):
    """MNIST images (HWC numpy, [0,1]) converted to NCHW float tensors."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # images: (N, 28, 28, 1) float32  →  tensor (N, 1, 28, 28)
        self.x = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        # labels: integer class indices (not one-hot) for CrossEntropyLoss
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MNISTConvNet(nn.Module):
    """PyTorch port of the Keras convnet: Conv→Pool→Conv→Pool→Flatten→Drop→Linear."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        # After Conv(3x3) on 28→26, MaxPool(2)→13, Conv(3x3)→11, MaxPool(2)→5
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 5 * 5, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_mnist_numpy():
    """Return (x_train, y_train, x_test, y_test) as numpy arrays via keras or torchvision."""
    try:
        import keras as _keras
        (xt, yt), (xe, ye) = _keras.datasets.mnist.load_data()
        xt = np.expand_dims(xt.astype("float32") / 255, -1)
        xe = np.expand_dims(xe.astype("float32") / 255, -1)
        return xt, yt, xe, ye
    except Exception:
        from torchvision import datasets, transforms
        _t = transforms.ToTensor()
        _tr = datasets.MNIST(root="/tmp/mnist", train=True,  download=True, transform=_t)
        _te = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=_t)
        xt = _tr.data.numpy().astype("float32")[:, :, :, None] / 255
        yt = _tr.targets.numpy()
        xe = _te.data.numpy().astype("float32")[:, :, :, None] / 255
        ye = _te.targets.numpy()
        return xt, yt, xe, ye


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate the model. Override __init__ kwargs via config['model_kwargs']."""
    kwargs = config.get("model_kwargs", {})
    return MNISTConvNet(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for 'train', 'val', or 'test'.
    MNIST is loaded via keras.datasets (fallback: torchvision).
    Client-local batch_size and num_workers are read from config['local'].
    """
    local       = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 128))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    x_train, y_train, x_test, y_test = _load_mnist_numpy()

    if split == "test":
        ds      = MNISTDataset(x_test, y_test)
        shuffle = False
    else:
        val_ratio = config.get("val_ratio", 0.1)
        n_val     = max(1, int(len(x_train) * val_ratio))
        n_train   = len(x_train) - n_val
        full_ds   = MNISTDataset(x_train, y_train)
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.get("seed", 42)),
        )
        ds      = train_ds if split == "train" else val_ds
        shuffle = split == "train"

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
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
    Perform one forward (+ optionally backward) step.
    If optimizer is None (preflight forward-only check), skip backward.
    """
    local   = config.get("local", {})
    use_amp = local.get("use_amp", False)
    device  = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch   = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        inputs  = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("label", batch.get("y", batch.get("target")))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    with torch.autocast(device_type=device.type if hasattr(device, "type") else str(device),
                        enabled=use_amp):
        outputs   = model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss      = criterion(outputs, targets)


    return loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import keras
    from keras import layers as keras_layers
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

try:
    import torchvision
    import torchvision.transforms as transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# Model / data parameters (preserved from original)
num_classes = 10
input_shape = (28, 28, 1)


class MNISTConvNet(nn.Module):
    """
    PyTorch equivalent of the Keras Sequential MNIST convnet.
    Original architecture by fchollet (2015/06/19, last modified 2020/04/21).
    Achieves ~99% test accuracy on MNIST.

    Architecture (mirroring original exactly):
        Conv2D(32, 3x3, relu) -> MaxPool2D(2x2)
        Conv2D(64, 3x3, relu) -> MaxPool2D(2x2)
        Flatten -> Dropout(0.5) -> Dense(num_classes, softmax)

    Spatial trace on 28x28 input:
        28 -[conv 3x3]-> 26 -[pool 2x2]-> 13
        13 -[conv 3x3]-> 11 -[pool 2x2]->  5
        Flattened: 64 * 5 * 5 = 1600
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1600, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)   # preserved from original Keras Dense(..., activation="softmax")


# ---------------------------------------------------------------------------
# FL interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the MNIST convnet."""
    model_kwargs = config.get("model_kwargs", {})
    return MNISTConvNet(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Data source priority:
      1. torchvision.datasets.MNIST (preferred)
      2. keras.datasets.mnist        (fallback if torchvision unavailable)
      3. Synthetic torch.randn/randint tensors — ONLY when
         config['allow_synthetic_data'] is True; otherwise raises FileNotFoundError.

    A 90/10 random_split is applied to the training portion of MNIST so that
    both "train" and "val" splits come from non-overlapping subsets.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    dataset = None

    # --- Attempt 1: torchvision MNIST (no automatic download) ---
    if HAS_TORCHVISION:
        try:
            transform = transforms.Compose([transforms.ToTensor()])
            full_ds = torchvision.datasets.MNIST(
                root=data_path, train=True, download=False, transform=transform
            )
            total = len(full_ds)
            val_size = int(0.1 * total)
            train_size = total - val_size
            train_ds, val_ds = random_split(
                full_ds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            dataset = train_ds if split == "train" else val_ds
        except Exception:
            dataset = None

    # --- Attempt 2: keras.datasets.mnist ---
    if dataset is None and HAS_KERAS:
        try:
            (x_train, y_train), _ = keras.datasets.mnist.load_data()
            # Replicate original pre-processing
            x_train = x_train.astype("float32") / 255
            x_train = np.expand_dims(x_train, -1)          # (N, 28, 28, 1)
            x_train = np.transpose(x_train, (0, 3, 1, 2))  # (N, 1, 28, 28) for PyTorch
            x_t = torch.from_numpy(x_train)
            y_t = torch.from_numpy(y_train.astype(np.int64))
            full_ds = TensorDataset(x_t, y_t)
            total = len(full_ds)
            val_size = int(0.1 * total)
            train_size = total - val_size
            train_ds, val_ds = random_split(
                full_ds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            dataset = train_ds if split == "train" else val_ds
        except Exception:
            dataset = None

    # --- Attempt 3: synthetic fallback (requires explicit opt-in) ---
    if dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                "MNIST dataset not found at data_path='{}'. "
                "Ensure torchvision (or keras) is installed and the data has been "
                "downloaded to that path, or set config['allow_synthetic_data'] = True "
                "to use randomly generated tensors for smoke-testing only.".format(data_path)
            )
        n_samples = 800 if split == "train" else 200
        x_syn = torch.randn(n_samples, 1, 28, 28)
        y_syn = torch.randint(0, num_classes, (n_samples,))
        dataset = TensorDataset(x_syn, y_syn)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=False,
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Run one forward pass and return the loss tensor (grad attached).

    Equivalent to Keras categorical_crossentropy on softmax outputs:
        loss = NLLLoss( log(softmax(logits)), y )
             = -sum( y_onehot * log(p) )

    NOTE: backward() and optimizer.step() are intentionally omitted;
    the FL runtime is responsible for both.
    """
    device = next(model.parameters()).device
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    output = model(x)  # (B, num_classes) softmax probabilities

    # clamp prevents log(0); equivalent to Keras categorical_crossentropy
    loss = F.nll_loss(torch.log(output.clamp(min=1e-10)), y)
    return loss  # grad is attached; caller handles backward + step
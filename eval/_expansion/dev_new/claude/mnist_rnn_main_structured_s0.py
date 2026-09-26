from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size=28, hidden_size=64, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, input):
        # Shape of input is (batch_size,1, 28, 28)
        # converting shape of input to (batch_size, 28, 28)
        # as required by RNN when batch_first is set True
        input = input.reshape(-1, 28, 28)
        output, hidden = self.rnn(input)

        # RNN output shape is (seq_len, batch, input_size)
        # Get last output of RNN
        output = output[:, -1, :]
        output = self.batchnorm(output)
        output = self.dropout1(output)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.dropout2(output)
        output = self.fc2(output)
        output = F.log_softmax(output, dim=1)
        return output


def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the Net model.

    Args:
        config: FL config dict. Supports key 'model_kwargs' (dict) whose
                contents are forwarded to Net.__init__ as keyword arguments.

    Returns:
        An initialised Net instance (on CPU; the FL runtime moves it to the
        target device as needed).
    """
    model_kwargs = config.get("model_kwargs", {})
    return Net(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the MNIST dataset.

    The full *training* MNIST split is divided 80/20 into a local train
    subset and a local validation subset using random_split.  The test
    MNIST split is never touched here so held-out evaluation remains clean.

    Args:
        config: FL config dict with optional keys:
            - data_path (str): root directory for the MNIST download
              (default ".").
            - local.batch_size (int): mini-batch size (default 16).
            - allow_synthetic_data (bool): when True a synthetic fallback
              dataset is used if real data cannot be found (default False).
        split: "train" or "val".

    Returns:
        A DataLoader over the requested subset.

    Raises:
        ValueError: if split is not "train" or "val".
        FileNotFoundError: if the MNIST dataset is absent and
            allow_synthetic_data is False.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # ------------------------------------------------------------------ #
    # Attempt to load the real MNIST dataset.                             #
    # ------------------------------------------------------------------ #
    full_dataset = None
    load_error = None
    try:
        full_dataset = datasets.MNIST(
            data_path, train=True, download=False, transform=transform
        )
    except Exception as exc:
        load_error = exc

    if full_dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset not found at '{data_path}' "
                f"(original error: {load_error}). "
                "Either provide a valid data_path containing the MNIST files "
                "or set config['allow_synthetic_data'] = True to use a "
                "synthetic stand-in for smoke-testing only."
            )

        # ---------------------------------------------------------------- #
        # Synthetic fallback — activated only when allow_synthetic_data    #
        # is explicitly True.                                               #
        # ---------------------------------------------------------------- #
        class _SyntheticMNIST(torch.utils.data.Dataset):
            """1 000 random MNIST-shaped samples for smoke-testing."""

            def __init__(self, size: int = 1000):
                self.data = torch.randn(size, 1, 28, 28)
                self.targets = torch.randint(0, 10, (size,))

            def __len__(self) -> int:
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]

        full_dataset = _SyntheticMNIST(size=1000)

    # ------------------------------------------------------------------ #
    # 80 / 20 train-val split.                                            #
    # ------------------------------------------------------------------ #
    total = len(full_dataset)
    val_size = max(1, int(round(0.2 * total)))
    train_size = total - val_size
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    if split == "train":
        return DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    else:  # "val"
        return DataLoader(val_subset, batch_size=batch_size, shuffle=False)


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); those calls must NOT appear here.

    Args:
        model: the Net instance in train() mode.
        batch: (data, target) tuple as yielded by the DataLoader.
        optimizer: the optimizer bound to model.parameters() (unused here
                   directly, but accepted per the FL interface contract).
        config: FL config dict (currently unused in the forward pass but
                kept for interface consistency and future use).

    Returns:
        Scalar loss tensor with requires_grad=True.
    """
    device = next(model.parameters()).device

    data, target = batch
    data = data.to(device)
    target = target.to(device)

    output = model(data)
    loss = F.nll_loss(output, target)
    return loss
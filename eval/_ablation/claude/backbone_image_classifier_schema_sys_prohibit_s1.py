# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os import path
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms


# ---------------------------------------------------------------------------
# Original model architecture preserved exactly
# ---------------------------------------------------------------------------

class Backbone(torch.nn.Module):
    """
    >>> Backbone()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Backbone(
      (l1): Linear(...)
      (l2): Linear(...)
    )
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))


# ---------------------------------------------------------------------------
# FL client API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the Backbone model.

    Args:
        config: FL config dict.  Constructor kwargs are read from
                config.get("model_kwargs", {}).

    Returns:
        An initialised Backbone instance.
    """
    model_kwargs = config.get("model_kwargs", {})
    return Backbone(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Tries to load the real MNIST dataset from *data_path*.  If the dataset
    cannot be found/loaded and ``config['allow_synthetic_data']`` is True,
    a small synthetic TensorDataset is used as a fallback.  If that flag is
    False (the default), a FileNotFoundError is raised instead so that the
    FL runtime is never silently trained on fake data.

    Args:
        config: FL config dict.
            - config.get("data_path", ".")          – root for MNIST files
            - config.get("local", {}).get("batch_size", 16) – batch size
            - config.get("allow_synthetic_data", False)     – synthetic gate
        split: "train" or "val".

    Returns:
        DataLoader for the requested split.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    # ---- attempt to load real MNIST ----------------------------------------
    try:
        if not _TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required to load MNIST images.")

        transform = transforms.ToTensor()
        full_dataset = MNIST(
            data_path,
            train=True,
            download=False,   # FL clients should have data pre-staged
            transform=transform,
        )

        total = len(full_dataset)
        val_size = max(1, int(0.1 * total))
        train_size = total - val_size

        train_subset, val_subset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        chosen = val_subset if split == "val" else train_subset
        return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))

    except Exception as exc:
        # ---- synthetic fallback (opt-in only) --------------------------------
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real MNIST dataset not found at '{data_path}' "
                f"(original error: {exc}). "
                "Pre-stage the data on this FL client, or set "
                "config['allow_synthetic_data'] = True to permit synthetic data."
            ) from exc

        # 1 000 synthetic 1×28×28 images with integer class labels in [0, 9]
        n_samples = 1000
        X = torch.randn(n_samples, 1, 28, 28)
        y = torch.randint(0, 10, (n_samples,))
        synthetic_dataset = torch.utils.data.TensorDataset(X, y)

        val_size = max(1, int(0.1 * n_samples))
        train_size = n_samples - val_size

        train_subset, val_subset = random_split(
            synthetic_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        chosen = val_subset if split == "val" else train_subset
        return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Execute one forward pass and return the loss tensor.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.

    Args:
        model:     The Backbone returned by build_model().
        batch:     A (images, labels) tuple from the DataLoader.
        optimizer: Provided by the FL runtime (unused here but kept in
                   signature for API conformance).
        config:    FL config dict (unused here; available for subclasses).

    Returns:
        Scalar cross-entropy loss tensor **with** a grad_fn attached.
    """
    device = next(model.parameters()).device

    x, y = batch
    x = x.to(device)
    y = y.to(device)

    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    # Return the live tensor — backward() is called by the FL runtime.
    return loss
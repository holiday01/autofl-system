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
"""FL client module derived from backbone_image_classifier.py.

Exposes build_model, build_dataloader, and train_step for use by an FL runtime.
The original LightningModule (LitClassifier) is preserved for reference but the
FL entry-points operate directly on the extracted Backbone nn.Module.
"""

from os import path
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


# ---------------------------------------------------------------------------
# Original model architecture – preserved exactly
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


class LitClassifier(LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(self, backbone: Optional[Backbone] = None, learning_rate: float = 0.0001):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        if backbone is None:
            backbone = Backbone()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class MyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        dataset = MNIST(DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def cli_main():
    cli = LightningCLI(
        LitClassifier, MyDataModule, seed_everything_default=1234, save_config_kwargs={"overwrite": True}, run=False
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    print(predictions[0])


# ---------------------------------------------------------------------------
# FL client entry-points
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the Backbone model.

    Args:
        config: FL config dict.  Constructor kwargs are read from
                config.get("model_kwargs", {}).  Supported key:
                  hidden_dim (int, default 128)

    Returns:
        A freshly initialised Backbone instance.
    """
    model_kwargs = config.get("model_kwargs", {})
    return Backbone(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Real data is attempted first.  Synthetic data is only used when
    config["allow_synthetic_data"] is explicitly True; otherwise a
    FileNotFoundError is raised so the FL runtime can surface the
    misconfiguration rather than silently training on noise.

    Args:
        config: FL config dict.  Relevant keys:
                  data_path (str)  – directory that contains the MNIST
                                     raw-data folder (default ".").
                  local.batch_size (int) – mini-batch size (default 16).
                  allow_synthetic_data (bool) – opt-in for synthetic
                                     fallback (default False).
        split:  "train" or "val".

    Returns:
        A DataLoader whose batches are (image_tensor, label_tensor) pairs.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    # ------------------------------------------------------------------ #
    # Attempt to load real MNIST data                                      #
    # ------------------------------------------------------------------ #
    real_dataset = None
    load_error: Optional[Exception] = None

    try:
        if not _TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is not installed; cannot apply transforms.ToTensor() "
                "to load MNIST images."
            )
        transform = transforms.ToTensor()
        # download=False: data must already be present at data_path.
        real_dataset = MNIST(data_path, train=True, download=False, transform=transform)
    except Exception as exc:
        load_error = exc

    if real_dataset is not None:
        # Produce a reproducible 90 / 10 train–val split.
        total = len(real_dataset)
        val_size = max(1, int(0.1 * total))
        train_size = total - val_size
        train_subset, val_subset = random_split(
            real_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        chosen = train_subset if split == "train" else val_subset
        return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))

    # ------------------------------------------------------------------ #
    # Real data unavailable – check the opt-in flag before using synth.  #
    # ------------------------------------------------------------------ #
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"MNIST dataset not found at '{data_path}'. "
            "Download the dataset and point config['data_path'] at the directory "
            "that contains the 'MNIST' raw-data folder, or set "
            "config['allow_synthetic_data'] = True to use synthetic data as a "
            "temporary fallback (not suitable for real training)."
        ) from load_error

    # Synthetic fallback (only reached when allow_synthetic_data=True).
    n_samples = 1000
    x_syn = torch.randn(n_samples, 1, 28, 28)
    y_syn = torch.randint(0, 10, (n_samples,))
    synthetic_dataset = TensorDataset(x_syn, y_syn)

    val_size = max(1, int(0.1 * n_samples))
    train_size = n_samples - val_size
    train_subset, val_subset = random_split(
        synthetic_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    chosen = train_subset if split == "train" else val_subset
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,  # noqa: ARG001  – held by FL runtime; not used here
    config: dict,  # noqa: ARG001  – reserved for future per-step config
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor (grad attached).

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.

    Args:
        model:     The Backbone instance returned by build_model().
        batch:     A (images, labels) tuple from build_dataloader().
        optimizer: Provided by the FL runtime (unused here).
        config:    FL config dict (reserved for future use).

    Returns:
        Cross-entropy loss tensor with requires_grad=True.
    """
    device = next(model.parameters()).device
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    # Deliberately no loss.backward() or optimizer.step() here.
    return loss


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
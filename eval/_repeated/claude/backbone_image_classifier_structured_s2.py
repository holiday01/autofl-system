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
from torch.utils.data import DataLoader, TensorDataset, random_split

from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


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
        # use forward for inference/predictions
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
        # self.hparams available because we called self.save_hyperparameters()
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


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()


# ---------------------------------------------------------------------------
# FL client API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the Backbone model.

    Args:
        config: FL config dict. Constructor kwargs are read from
                config.get("model_kwargs", {}) and forwarded to Backbone.

    Returns:
        An initialised Backbone instance.
    """
    model_kwargs = config.get("model_kwargs", {})
    return Backbone(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Attempts to load MNIST from *data_path*. If that fails and
    ``config["allow_synthetic_data"]`` is True, a synthetic TensorDataset is
    used instead. If real data is unavailable and the flag is False, a
    FileNotFoundError is raised — synthetic data is *never* used silently.

    Args:
        config: FL config dict. Relevant keys:
            - ``data_path``            (str,  default ".")
            - ``local.batch_size``     (int,  default 16)
            - ``allow_synthetic_data`` (bool, default False)
        split: ``"train"`` or ``"val"``.

    Returns:
        A configured DataLoader for the requested split.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    # ------------------------------------------------------------------ #
    # Attempt to load the real MNIST dataset (no automatic download).     #
    # ------------------------------------------------------------------ #
    real_data_error: Optional[Exception] = None

    if _TORCHVISION_AVAILABLE:
        try:
            full_dataset = MNIST(
                data_path,
                train=True,
                download=False,
                transform=transforms.ToTensor(),
            )
            n_total = len(full_dataset)
            n_train = int(0.9 * n_total)
            n_val = n_total - n_train
            train_subset, val_subset = random_split(
                full_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            chosen = train_subset if split == "train" else val_subset
            return DataLoader(
                chosen,
                batch_size=batch_size,
                shuffle=(split == "train"),
            )
        except Exception as exc:
            real_data_error = exc
    else:
        real_data_error = ImportError(
            "torchvision is not installed; cannot load MNIST."
        )

    # ------------------------------------------------------------------ #
    # Real data unavailable — honour the synthetic-data gate.             #
    # ------------------------------------------------------------------ #
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"MNIST dataset not found at '{data_path}'. "
            "Either provide a valid 'data_path' in config or set "
            "config['allow_synthetic_data'] = True to permit synthetic data."
        ) from real_data_error

    # Synthetic fallback: (N, 1, 28, 28) images + integer class labels.
    n_samples = 1000 if split == "train" else 200
    x_syn = torch.randn(n_samples, 1, 28, 28)
    y_syn = torch.randint(0, 10, (n_samples,))
    synthetic_dataset = TensorDataset(x_syn, y_syn)
    return DataLoader(
        synthetic_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,  # held by the FL runtime; not used here
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor (grad attached).

    The FL runtime is responsible for calling ``loss.backward()`` and
    ``optimizer.step()``; this function must NOT do either.

    Args:
        model:     The Backbone returned by ``build_model``.
        batch:     A ``(images, labels)`` tuple from the DataLoader.
        optimizer: Passed by the FL runtime (unused inside this function).
        config:    FL config dict (unused here but kept for API uniformity).

    Returns:
        Cross-entropy loss scalar tensor with ``requires_grad=True``.
    """
    device = next(model.parameters()).device
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    return loss  # grad attached; backward/step handled by the FL runtime
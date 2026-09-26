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
# FL Client Interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the Backbone model.

    Accepts optional constructor overrides via config['model_kwargs'], e.g.:
        {"model_kwargs": {"hidden_dim": 256}}
    """
    model_kwargs = config.get("model_kwargs", {})
    return Backbone(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split ('train' or 'val').

    Tries to load MNIST from config['data_path'] (no auto-download).
    Falls back to synthetic tensors only when config['allow_synthetic_data'] is True;
    otherwise raises FileNotFoundError so the caller knows data is missing.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    # ── Try to load the real dataset ─────────────────────────────────────────
    real_dataset = None
    load_exc: Optional[Exception] = None

    if not _TORCHVISION_AVAILABLE:
        load_exc = ImportError(
            "torchvision is not installed; cannot load MNIST images."
        )
    else:
        try:
            # download=False: the FL runtime controls data provisioning.
            real_dataset = MNIST(
                data_path,
                train=True,
                download=False,
                transform=transforms.ToTensor(),
            )
        except Exception as exc:  # noqa: BLE001
            load_exc = exc

    # ── Synthetic fallback gate ───────────────────────────────────────────────
    if real_dataset is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"MNIST dataset not found at '{data_path}'. "
                "Either supply preprocessed MNIST data at that path, or set "
                "config['allow_synthetic_data'] = True to use random tensors "
                "(for smoke-testing only)."
            ) from load_exc

        # Synthetic stand-in: (N, 1, 28, 28) float images + long labels
        n_samples = 1000 if split == "train" else 200
        x = torch.randn(n_samples, 1, 28, 28)
        y = torch.randint(0, 10, (n_samples,))
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))

    # ── Train / val split from the real dataset ───────────────────────────────
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


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,  # noqa: ARG001 — held by the FL runtime; not used here
    config: dict,  # noqa: ARG001 — available for loss kwargs if needed
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor (grad attached).

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.
    """
    device = next(model.parameters()).device
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    return loss  # grad graph intact; no backward / step here
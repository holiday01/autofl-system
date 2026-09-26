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
"""MNIST autoencoder — FL client module.

Exposes:
    build_model(config)              -> torch.nn.Module
    build_dataloader(config, split)  -> DataLoader
    train_step(model, batch, optimizer, config) -> torch.Tensor
"""

from os import path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from lightning.pytorch import LightningDataModule, LightningModule, Trainer, callbacks, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision import transforms
    from torchvision.utils import save_image

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


# ---------------------------------------------------------------------------
# Original classes preserved verbatim
# ---------------------------------------------------------------------------

class ImageSampler(callbacks.Callback):
    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        value_range: Optional[tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.value_range = value_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def _to_grid(self, images):
        return torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            value_range=self.value_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not _TORCHVISION_AVAILABLE:
            return

        images, _ = next(iter(DataLoader(trainer.datamodule.mnist_val, batch_size=self.num_samples)))
        images_flattened = images.view(images.size(0), -1)

        with torch.no_grad():
            pl_module.eval()
            images_generated = pl_module(images_flattened.to(pl_module.device))
            pl_module.train()

        if trainer.current_epoch == 0:
            save_image(self._to_grid(images), f"grid_ori_{trainer.current_epoch}.png")
        save_image(self._to_grid(images_generated.reshape(images.shape)), f"grid_generated_{trainer.current_epoch}.png")


class LitAutoEncoder(LightningModule):
    """
    >>> LitAutoEncoder()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

    def __init__(self, hidden_dim: int = 64, learning_rate=10e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.decoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 28 * 28))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def _prepare_batch(self, batch):
        x, _ = batch
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(x, self(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss


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
        LitAutoEncoder,
        MyDataModule,
        seed_everything_default=1234,
        run=False,
        trainer_defaults={"callbacks": ImageSampler(), "max_epochs": 10},
        save_config_kwargs={"overwrite": True},
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    print(predictions[0])


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()


# ---------------------------------------------------------------------------
# Extracted pure nn.Module (identical architecture, no Lightning dependency)
# ---------------------------------------------------------------------------

class AutoEncoder(nn.Module):
    """Pure-PyTorch autoencoder extracted from LitAutoEncoder for FL use."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# FL interface
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the AutoEncoder.

    Args:
        config: FL config dict.  Constructor kwargs are read from
                config.get("model_kwargs", {}).  Supported keys:
                    hidden_dim (int, default 64)
    """
    kwargs = config.get("model_kwargs", {})
    return AutoEncoder(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Tries to load the real MNIST dataset from config["data_path"].
    Falls back to synthetic data only when config["allow_synthetic_data"] is
    explicitly True; otherwise raises FileNotFoundError.

    Args:
        config: FL config dict with optional keys:
                    data_path            (str,  default ".")
                    local.batch_size     (int,  default 16)
                    allow_synthetic_data (bool, default False)
        split:  "train" or "val"
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")

    # ------------------------------------------------------------------
    # Attempt to load real MNIST
    # ------------------------------------------------------------------
    _load_error: Optional[Exception] = None
    try:
        if not _TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required to load MNIST but is not installed."
            )
        transform = transforms.ToTensor()
        full_dataset = MNIST(
            data_path, train=True, download=True, transform=transform
        )
        # Preserve original 55 000 / 5 000 split
        n_total = len(full_dataset)
        n_val = min(5000, n_total)
        n_train = n_total - n_val
        train_subset, val_subset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        subset = train_subset if split == "train" else val_subset
        return DataLoader(subset, batch_size=batch_size, shuffle=(split == "train"))
    except Exception as exc:  # noqa: BLE001
        _load_error = exc

    # ------------------------------------------------------------------
    # Real data unavailable — honour the synthetic-data flag
    # ------------------------------------------------------------------
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"MNIST dataset could not be loaded from data_path='{data_path}' "
            f"(underlying error: {_load_error}). "
            "Either supply a valid data_path or set config['allow_synthetic_data'] = True "
            "to fall back to synthetic data."
        ) from _load_error

    # Synthetic fallback: (N, 1, 28, 28) images + integer labels
    n_samples = 55000 if split == "train" else 5000
    x_synth = torch.randn(n_samples, 1, 28, 28)
    y_synth = torch.randint(0, 10, (n_samples,))
    synthetic_ds = TensorDataset(x_synth, y_synth)
    return DataLoader(synthetic_ds, batch_size=batch_size, shuffle=(split == "train"))


def train_step(
    model: nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the MSE reconstruction loss.

    The FL runtime is responsible for calling loss.backward() and
    optimizer.step(); this function must NOT do either.

    Args:
        model:     The AutoEncoder returned by build_model().
        batch:     A (images, labels) tuple as yielded by build_dataloader().
        optimizer: Provided by the FL runtime (unused here but part of the
                   standard FL client signature).
        config:    FL config dict (unused in this step but kept for uniformity).

    Returns:
        Scalar loss tensor with grad_fn attached.
    """
    device = next(model.parameters()).device

    x, _ = batch
    x = x.to(device)
    # Flatten spatial dims: (B, 1, 28, 28) -> (B, 784)
    x = x.view(x.size(0), -1)

    x_hat = model(x)
    loss = F.mse_loss(x_hat, x)
    return loss
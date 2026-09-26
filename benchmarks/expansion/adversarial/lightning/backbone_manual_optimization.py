# ============================================================================
#  AutoFL benchmark expansion — ADVERSARIAL MUTANT A3: backbone_manual_optimization
#  Parent   : lightning/backbone_image_classifier.py (sha256 a34c1eda8065de16185466d8abba545e770567a639c0f4e49b9f630523b2db1b); the parent is a dev-set script,
#             so this mutant leaks nothing about the holdout set and cannot exist in any training corpus.
#  Change   : `self.automatic_optimization = False`; training_step does `opt = self.optimizers(); opt.zero_grad(); self.manual_backward(loss); opt.step(); self.lr_schedulers().step()`; configure_optimizers returns ([Adam], [StepLR])
#  Stresses : I2 (+I1): the backward pass is hidden behind `self.manual_backward`, the optimizer is stepped by the module, and the LR scheduler is stepped per batch
#  Correct conversion : train_step returns `loss` only; no manual_backward, no opt.step(), no scheduler.step()
#  Plausible-but-wrong: translating `manual_backward` into `loss.backward()` + `optimizer.step()` inside train_step
#  Notes    : isolates the manual-optimization mechanism from GAN semantics (the dev-set Lightning GAN already combines both)
#  Diff vs parent: 10 changed lines (+9 / -1), unified diff with zero context below;
#  the header block you are reading is NOT part of the diff (the body after it is parent + diff).
#  --- diff ---
#  --- lightning/backbone_image_classifier.py
#  +++ benchmarks/expansion/adversarial/lightning/backbone_manual_optimization.py
#  @@ -68,0 +69 @@
#  +        self.automatic_optimization = False
#  @@ -80,0 +82,5 @@
#  +        opt = self.optimizers()
#  +        opt.zero_grad()
#  +        self.manual_backward(loss)
#  +        opt.step()
#  +        self.lr_schedulers().step()
#  @@ -102 +108,3 @@
#  -        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
#  +        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
#  +        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
#  +        return [optimizer], [scheduler]
#  --- end diff ---
# ============================================================================
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
"""MNIST backbone image classifier example.

To run: python backbone_image_classifier.py --trainer.max_epochs=50

"""

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
        self.automatic_optimization = False
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
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.lr_schedulers().step()
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        return [optimizer], [scheduler]


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

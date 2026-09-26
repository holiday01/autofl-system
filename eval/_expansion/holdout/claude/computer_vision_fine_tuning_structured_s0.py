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

import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.callbacks.finetuning import BaseFinetuning
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.utilities.model_helpers import get_torchvision_model

log = logging.getLogger(__name__)
DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"


# ---------------------------------------------------------------------------
# Original classes preserved exactly
# ---------------------------------------------------------------------------

class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer):
        if epoch == self.milestones[0]:
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:], optimizer=optimizer, train_bn=self.train_bn
            )
        elif epoch == self.milestones[1]:
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5], optimizer=optimizer, train_bn=self.train_bn
            )


class CatDogImageDataModule(LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8):
        super().__init__()
        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    def prepare_data(self):
        download_and_extract_archive(url=DATA_URL, download_root=self._dl_path, remove_finished=True)

    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("cats_and_dogs_filtered")

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), self.normalize_transform])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        if train:
            dataset = self.create_dataset(self.data_path.joinpath("train"), self.train_transform)
        else:
            dataset = self.create_dataset(self.data_path.joinpath("validation"), self.valid_transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        log.info("Training data loaded.")
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info("Validation data loaded.")
        return self.__dataloader(train=False)


class TransferLearningModel(LightningModule):
    def __init__(
        self,
        backbone: str = "resnet50",
        train_bn: bool = False,
        milestones: tuple = (2, 4),
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        self.__build_model()

        self.train_acc = Accuracy(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.save_hyperparameters()

    def __build_model(self):
        backbone = get_torchvision_model(self.backbone, weights="DEFAULT")
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)
        _fc_layers = [nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, 1)]
        self.fc = nn.Sequential(*_fc_layers)
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.fc(x)

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
        train_loss = self.loss(y_logits, y_true)
        self.log("train_acc", self.train_acc(y_scores, y_true.int()), prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)
        self.log("val_acc", self.valid_acc(y_scores, y_true.int()), prog_bar=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer], [scheduler]


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MilestonesFinetuning, "finetuning")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("finetuning.milestones", "model.milestones")
        parser.link_arguments("finetuning.train_bn", "model.train_bn")
        parser.set_defaults({
            "trainer.max_epochs": 15,
            "trainer.enable_model_summary": False,
            "trainer.num_sanity_val_steps": 0,
        })


def cli_main():
    MyLightningCLI(TransferLearningModel, CatDogImageDataModule, seed_everything_default=1234)


# ---------------------------------------------------------------------------
# Pure nn.Module extracted from TransferLearningModel for FL use
# ---------------------------------------------------------------------------

class TransferLearningNet(nn.Module):
    """Standalone nn.Module equivalent of TransferLearningModel for FL clients."""

    def __init__(self, backbone: str = "resnet50") -> None:
        super().__init__()
        backbone_model = get_torchvision_model(backbone, weights="DEFAULT")
        _layers = list(backbone_model.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)
        _fc_layers = [nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, 1)]
        self.fc = nn.Sequential(*_fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return TransferLearningNet.

    config keys consumed:
        model_kwargs.backbone  (str, default "resnet50")
    """
    model_kwargs = config.get("model_kwargs", {})
    backbone = model_kwargs.get("backbone", "resnet50")
    return TransferLearningNet(backbone=backbone)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Searches for an ImageFolder-compatible dataset under data_path in this
    priority order:
        1. <data_path>/cats_and_dogs_filtered/train
        2. <data_path>/train
        3. <data_path>

    A fixed-seed random_split (80 / 20) is applied so that both calls with
    split="train" and split="val" refer to non-overlapping subsets of the
    same full dataset.

    If no real data is found and config["allow_synthetic_data"] is False,
    a FileNotFoundError is raised.  Set it to True to fall back to random
    tensors (useful for smoke-testing the FL runtime).

    config keys consumed:
        local.batch_size          (int,  default 16)
        data_path                 (str,  default ".")
        allow_synthetic_data      (bool, default False)
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    batch_size: int = config.get("local", {}).get("batch_size", 16)
    data_path = Path(config.get("data_path", "."))

    _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    _train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize,
    ])
    _val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _normalize,
    ])

    chosen_transform = _train_transform if split == "train" else _val_transform

    # Locate the dataset root
    _candidates = [
        data_path / "cats_and_dogs_filtered" / "train",
        data_path / "train",
        data_path,
    ]
    found_root: Union[Path, None] = None
    for candidate in _candidates:
        if candidate.exists() and candidate.is_dir():
            try:
                # Quick check: ImageFolder requires at least one class sub-directory
                subdirs = [p for p in candidate.iterdir() if p.is_dir()]
                if subdirs:
                    found_root = candidate
                    break
            except PermissionError:
                continue

    if found_root is None:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"No ImageFolder-compatible dataset found under '{data_path}' "
                "(tried: cats_and_dogs_filtered/train, train/, and the root itself). "
                "Point config['data_path'] at the correct directory, or set "
                "config['allow_synthetic_data'] = True to run on synthetic tensors."
            )
        # --- Synthetic data fallback (smoke-test only) ---
        _n = 100
        _x = torch.randn(_n, 3, 224, 224)
        _y = torch.randint(0, 2, (_n,)).float()
        _full_ds = TensorDataset(_x, _y)
        _train_n = int(0.8 * _n)
        _val_n = _n - _train_n
        _gen = torch.Generator().manual_seed(42)
        _train_ds, _val_ds = random_split(_full_ds, [_train_n, _val_n], generator=_gen)
        _ds = _train_ds if split == "train" else _val_ds
        return DataLoader(_ds, batch_size=batch_size, shuffle=(split == "train"))

    # --- Real data path ---
    # Load the full ImageFolder with the transform appropriate for this split
    # so that augmentation is applied only to training indices and plain
    # resizing+normalisation is applied to validation indices.
    full_ds = ImageFolder(root=str(found_root), transform=chosen_transform)
    total = len(full_ds)
    train_n = int(0.8 * total)
    val_n = total - train_n
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [train_n, val_n], generator=generator)
    ds = train_ds if split == "train" else val_ds
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer,  # held by FL runtime; not stepped here
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for loss.backward() and optimizer.step().
    This function must NOT call either.
    """
    device = next(model.parameters()).device
    x, y = batch
    x = x.to(device)
    # Labels arrive as class indices (LongTensor from ImageFolder / synthetic).
    # The model outputs a single logit → binary cross-entropy expects Float.
    y = y.to(device).view(-1, 1).float()
    logits = model(x)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    return loss


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
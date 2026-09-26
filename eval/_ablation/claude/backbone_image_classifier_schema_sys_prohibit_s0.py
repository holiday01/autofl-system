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


# ---------------------------------------------------------------------------
# FL client API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and return the Backbone model.

    Args:
        config: FL config dict. Supports key ``model_kwargs`` (dict) passed
                directly to :class:`Backbone`, e.g.
                ``{"model_kwargs": {"hidden_dim": 256}}``.

    Returns:
        An initialised :class:`Backbone` instance.
    """
    model_kwargs = config.get("model_kwargs", {})
    return Backbone(**model_kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Real data is loaded from the MNIST dataset located at
    ``config["data_path"]`` (default ``"."``).  When the dataset cannot be
    found and ``config["allow_synthetic_data"]`` is ``True``, a small
    synthetic :class:`~torch.utils.data.TensorDataset` (1 000 random
    28×28 images, 10 classes) is used instead.  If the dataset is missing
    and ``allow_synthetic_data`` is ``False`` (the default), a
    :class:`FileNotFoundError` is raised — synthetic data is *never* used
    silently.

    Args:
        config: FL config dict.  Recognised keys:

            * ``"data_path"`` (str, default ``"."``) – root directory that
              contains (or will contain) the MNIST folder.
            * ``"local"`` (dict) – sub-dict; reads ``"batch_size"``
              (int, default ``16``).
            * ``"allow_synthetic_data"`` (bool, default ``False``) – gate
              for the synthetic fallback.

        split: ``"train"`` or ``"val"``.

    Returns:
        A :class:`~torch.utils.data.DataLoader` over the requested split.

    Raises:
        ValueError: If *split* is not ``"train"`` or ``"val"``.
        FileNotFoundError: If real data is unavailable and
            ``allow_synthetic_data`` is ``False``.
    """
    if split not in ("train", "val"):
        raise ValueError(f"Unknown split {split!r}. Expected 'train' or 'val'.")

    batch_size: int = config.get("local", {}).get("batch_size", 16)
    data_path: str = config.get("data_path", ".")

    real_data_error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Attempt to load real MNIST data
    # ------------------------------------------------------------------
    if _TORCHVISION_AVAILABLE:
        try:
            transform = transforms.ToTensor()
            full_dataset = MNIST(
                data_path, train=True, download=True, transform=transform
            )

            total = len(full_dataset)
            val_size = max(1, int(0.1 * total))
            train_size = total - val_size

            train_subset, val_subset = random_split(
                full_dataset,
                [train_size, val_size],
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
            "torchvision is not available; cannot load MNIST with transforms."
        )

    # ------------------------------------------------------------------
    # Real data unavailable — honour the allow_synthetic_data gate
    # ------------------------------------------------------------------
    if not config.get("allow_synthetic_data", False):
        raise FileNotFoundError(
            f"MNIST dataset not found at '{data_path}' and "
            "'allow_synthetic_data' is False.  "
            "Either set config['data_path'] to a directory that contains "
            "the MNIST files, or set config['allow_synthetic_data'] = True "
            "to use randomly generated placeholder data."
        ) from real_data_error

    # Synthetic fallback (only reached when allow_synthetic_data is True)
    n_samples = 1000
    X_synth = torch.randn(n_samples, 1, 28, 28)
    y_synth = torch.randint(0, 10, (n_samples,))
    synth_dataset = torch.utils.data.TensorDataset(X_synth, y_synth)

    val_size_s = max(1, int(0.1 * n_samples))
    train_size_s = n_samples - val_size_s

    train_subset_s, val_subset_s = random_split(
        synth_dataset,
        [train_size_s, val_size_s],
        generator=torch.Generator().manual_seed(42),
    )

    chosen_s = train_subset_s if split == "train" else val_subset_s
    return DataLoader(
        chosen_s,
        batch_size=batch_size,
        shuffle=(split == "train"),
    )


def train_step(
    model: torch.nn.Module,
    batch,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor.

    The FL runtime is responsible for calling ``loss.backward()`` and
    ``optimizer.step()``; this function must **not** do either.

    Args:
        model:     The :class:`Backbone` returned by :func:`build_model`.
        batch:     A ``(images, labels)`` tuple from the DataLoader.
        optimizer: Provided by the FL runtime (unused here but kept for
                   API consistency).
        config:    FL config dict (currently unused in this step).

    Returns:
        The scalar cross-entropy loss **with its grad_fn intact**.
    """
    device = next(model.parameters()).device

    x, y = batch
    x = x.to(device)
    y = y.to(device)

    y_hat = model(x)
    loss = F.cross_entropy(y_hat, y)
    return loss
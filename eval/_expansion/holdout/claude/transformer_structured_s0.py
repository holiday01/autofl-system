import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

import lightning as L
from lightning.pytorch.demos import Transformer, WikiText2


# ---------------------------------------------------------------------------
# Model — underlying architecture extracted from the original LightningModule.
# The LightningModule wrapper is intentionally dropped; the FL runtime owns
# the training loop.
# ---------------------------------------------------------------------------

class LanguageModel(nn.Module):
    """Transformer-based language model extracted from the original LightningModule."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.model(input, target)


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the LanguageModel.

    Config keys consumed:
        model_kwargs.vocab_size  – vocabulary size (default: 33_278, the
                                   WikiText2 vocab size used by the original
                                   script's Transformer demo).
    """
    model_kwargs = config.get("model_kwargs", {})
    vocab_size = model_kwargs.get("vocab_size", 33_278)
    return LanguageModel(vocab_size=vocab_size)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for *split* ('train' or 'val').

    Config keys consumed:
        local.batch_size          – mini-batch size (default: 16).
        data_path                 – directory that contains (or will receive)
                                    the WikiText2 data (default: '.').
        allow_synthetic_data      – if True and the real dataset is absent,
                                    fall back to random token tensors.
                                    If False (default) and the dataset is
                                    absent, raise FileNotFoundError.
        model_kwargs.vocab_size   – used only for synthetic fallback vocab
                                    range (default: 33_278).
    """
    if split not in ("train", "val"):
        raise ValueError(f"Unknown split '{split}'. Expected 'train' or 'val'.")

    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path  = config.get("data_path", ".")

    # ------------------------------------------------------------------
    # Attempt to load the real WikiText2 dataset.
    # WikiText2 from lightning.pytorch.demos accepts an optional positional
    # data_dir; fall back to the no-arg form for older Lightning versions.
    # ------------------------------------------------------------------
    dataset = None
    _load_error: Exception | None = None

    for _args in [(data_path,), ()]:
        try:
            dataset = WikiText2(*_args)
            break
        except TypeError:
            # Signature doesn't accept positional data_dir — retry without it.
            continue
        except Exception as exc:
            _load_error = exc
            break

    if dataset is None:
        # WikiText2 was unavailable (download failed, path wrong, etc.).
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"WikiText2 dataset could not be loaded from '{data_path}'. "
                "Ensure the data is present or set config['allow_synthetic_data'] = True "
                "to use synthetic token data instead."
            ) from _load_error

        # ----- Synthetic fallback (only when explicitly allowed) -----
        vocab_size  = config.get("model_kwargs", {}).get("vocab_size", 33_278)
        num_samples = 1_000
        block_size  = 35  # matches WikiText2 default block_size
        inputs  = torch.randint(0, vocab_size, (num_samples, block_size))
        targets = torch.randint(0, vocab_size, (num_samples, block_size))
        full_ds = TensorDataset(inputs, targets)

        val_size   = max(1, int(0.2 * num_samples))
        train_size = num_samples - val_size
        train_ds, val_ds = random_split(full_ds, [train_size, val_size])

        chosen_ds = train_ds if split == "train" else val_ds
        return DataLoader(chosen_ds, batch_size=batch_size, shuffle=(split == "train"))

    # ------------------------------------------------------------------
    # Real dataset: carve out a train / val split.
    # The original script used [n-4000, 2000, 2000]; here we fold the test
    # portion into train to keep a clean two-way split for FL rounds.
    # ------------------------------------------------------------------
    n        = len(dataset)
    val_size  = max(1, min(2_000, int(0.2 * n)))
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    chosen_ds = train_ds if split == "train" else val_ds
    return DataLoader(
        chosen_ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,          # noqa: ARG001 — owned by the FL runtime
    config: dict,       # noqa: ARG001 — available for future extensions
) -> torch.Tensor:
    """Run one forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for loss.backward() and optimizer.step();
    do NOT call them here.
    """
    device = next(model.parameters()).device

    input, target = batch
    input  = input.to(device)
    target = target.to(device)

    output = model(input, target)                    # forward only
    loss   = F.nll_loss(output, target.view(-1))     # grad graph intact
    return loss                                      # NO backward / step
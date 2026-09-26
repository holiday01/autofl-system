"""
Auto-generated FL client module.
Original script: character-level LSTM seq2seq (English → French).

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class Seq2SeqCharDataset(Dataset):
    """Character-level English→French dataset with one-hot encoder inputs/decoder inputs
    and integer class-index decoder targets (teacher-forcing offset by one timestep)."""

    def __init__(self, data_path: str, num_samples: int = 10000):
        input_texts, target_texts = [], []
        input_characters, target_characters = set(), set()

        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        for line in lines[: min(num_samples, len(lines) - 1)]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            input_text, target_text = parts[0], parts[1]
            target_text = "\t" + target_text + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)
            input_characters.update(input_text)
            target_characters.update(target_text)

        input_characters = sorted(input_characters)
        target_characters = sorted(target_characters)
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max(len(t) for t in input_texts)
        self.max_decoder_seq_length = max(len(t) for t in target_texts)

        input_token_index = {c: i for i, c in enumerate(input_characters)}
        target_token_index = {c: i for i, c in enumerate(target_characters)}

        n = len(input_texts)
        enc_len = self.max_encoder_seq_length
        dec_len = self.max_decoder_seq_length
        num_enc = self.num_encoder_tokens
        num_dec = self.num_decoder_tokens

        encoder_input_data = np.zeros((n, enc_len, num_enc), dtype="float32")
        decoder_input_data = np.zeros((n, dec_len, num_dec), dtype="float32")
        # Store decoder targets as class indices rather than one-hot for CrossEntropyLoss
        decoder_target_data = np.zeros((n, dec_len), dtype="int64")

        space_enc = input_token_index.get(" ", 0)
        space_dec = target_token_index.get(" ", 0)

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            t = 0
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.0
            encoder_input_data[i, t + 1 :, space_enc] = 1.0

            for t, char in enumerate(target_text):
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    # decoder_target is decoder_input offset forward by one timestep
                    decoder_target_data[i, t - 1] = target_token_index[char]
            decoder_input_data[i, t + 1 :, space_dec] = 1.0
            decoder_target_data[i, t:] = space_dec

        self.encoder_inputs = torch.from_numpy(encoder_input_data)
        self.decoder_inputs = torch.from_numpy(decoder_input_data)
        self.decoder_targets = torch.from_numpy(decoder_target_data)

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        return (
            self.encoder_inputs[idx],
            self.decoder_inputs[idx],
            self.decoder_targets[idx],
        )


class Seq2SeqLSTM(nn.Module):
    """Character-level LSTM encoder-decoder for sequence-to-sequence translation."""

    def __init__(
        self,
        num_encoder_tokens: int,
        num_decoder_tokens: int,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.encoder_lstm = nn.LSTM(num_encoder_tokens, latent_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(num_decoder_tokens, latent_dim, batch_first=True)
        self.decoder_dense = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        _, (h, c) = self.encoder_lstm(encoder_input)
        decoder_output, _ = self.decoder_lstm(decoder_input, (h, c))
        return self.decoder_dense(decoder_output)  # (batch, dec_seq_len, num_decoder_tokens)


# ── helpers ─────────────────────────────────────────────────────────────

class _SyntheticSeq2SeqDataset(Dataset):
    def __init__(
        self,
        n: int = 200,
        num_encoder_tokens: int = 71,
        num_decoder_tokens: int = 93,
        max_encoder_seq_length: int = 16,
        max_decoder_seq_length: int = 59,
    ):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.encoder_inputs = torch.randn(n, max_encoder_seq_length, num_encoder_tokens)
        self.decoder_inputs = torch.randn(n, max_decoder_seq_length, num_decoder_tokens)
        self.decoder_targets = torch.randint(0, num_decoder_tokens, (n, max_decoder_seq_length))

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        return (
            self.encoder_inputs[idx],
            self.decoder_inputs[idx],
            self.decoder_targets[idx],
        )


# ── FL Interface ─────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return Seq2SeqLSTM(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    data_path   = config.get("data_path", "fra.txt")
    num_samples = config.get("num_samples", 10000)
    val_ratio   = config.get("val_ratio", 0.1)

    if os.path.isfile(data_path):
        full_dataset = Seq2SeqCharDataset(data_path=data_path, num_samples=num_samples)
    else:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Data file not found: {data_path!r}. "
                "Set config['allow_synthetic_data'] = True to use synthetic data."
            )
        dataset_kwargs = config.get("dataset_kwargs", {})
        full_dataset = _SyntheticSeq2SeqDataset(**dataset_kwargs)

    n_val = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.get("seed", 42)),
    )
    ds = train_ds if split == "train" else val_ds
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def train_step(
    model: nn.Module,
    batch: tuple | list,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        encoder_input, decoder_input, decoder_target = batch[0], batch[1], batch[2]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        encoder_input  = batch.get("encoder_input", batch.get("x_enc"))
        decoder_input  = batch.get("decoder_input", batch.get("x_dec"))
        decoder_target = batch.get("decoder_target", batch.get("y"))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    logits = model(encoder_input, decoder_input)
    # logits: (batch, dec_seq_len, num_decoder_tokens)
    # CrossEntropyLoss expects (N, C) with integer targets (N,) after flattening
    _, dec_seq_len, num_dec_tok = logits.shape
    criterion = nn.CrossEntropyLoss()
    loss = criterion(
        logits.reshape(-1, num_dec_tok),
        decoder_target.reshape(-1),
    )
    return loss
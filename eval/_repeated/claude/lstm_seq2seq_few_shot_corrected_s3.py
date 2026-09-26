"""
Auto-generated FL client module.
Original script: character-level LSTM seq2seq translation (Keras → PyTorch).

Exposes:
  build_model(config)               -> nn.Module
  build_dataloader(config, split)   -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT (read carefully before copying this pattern):
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.

  model_kwargs must include: num_encoder_tokens, num_decoder_tokens.
  These are dataset-dependent; compute from Seq2SeqCharDataset attributes
  and pass them explicitly in config["model_kwargs"].
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ── Dataset ──────────────────────────────────────────────────────────────

class Seq2SeqCharDataset(Dataset):
    """
    Character-level English→French dataset (fra-eng Anki format).
    Returns (encoder_input, decoder_input, decoder_target) as float32 tensors
    of shape (max_enc_len, num_enc_tokens), (max_dec_len, num_dec_tokens),
    (max_dec_len, num_dec_tokens).
    """

    def __init__(self, data_path: str = "fra.txt", num_samples: int = 10000):
        input_texts, target_texts = [], []
        input_chars, target_chars = set(), set()

        if os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
        else:
            # synthetic fallback for testing (single pair repeated)
            lines = ["Hello.\tBonjour.\tCC-BY 2.0"] * min(num_samples, 200)

        for line in lines[: min(num_samples, len(lines) - 1)]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            inp, tgt = parts[0], "\t" + parts[1] + "\n"
            input_texts.append(inp)
            target_texts.append(tgt)
            input_chars.update(inp)
            target_chars.update(tgt)

        input_chars = sorted(input_chars)
        target_chars = sorted(target_chars)
        self.input_token_index  = {c: i for i, c in enumerate(input_chars)}
        self.target_token_index = {c: i for i, c in enumerate(target_chars)}
        self.num_encoder_tokens  = len(input_chars)
        self.num_decoder_tokens  = len(target_chars)
        self.max_encoder_seq_length = max(len(t) for t in input_texts)
        self.max_decoder_seq_length = max(len(t) for t in target_texts)

        n = len(input_texts)
        enc = np.zeros((n, self.max_encoder_seq_length, self.num_encoder_tokens), dtype=np.float32)
        dec_in  = np.zeros((n, self.max_decoder_seq_length, self.num_decoder_tokens), dtype=np.float32)
        dec_out = np.zeros((n, self.max_decoder_seq_length, self.num_decoder_tokens), dtype=np.float32)

        space_enc = self.input_token_index.get(" ", 0)
        space_dec = self.target_token_index.get(" ", 0)

        for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
            t = 0
            for t, c in enumerate(inp):
                enc[i, t, self.input_token_index[c]] = 1.0
            enc[i, t + 1:, space_enc] = 1.0
            for t, c in enumerate(tgt):
                dec_in[i, t, self.target_token_index[c]] = 1.0
                if t > 0:
                    dec_out[i, t - 1, self.target_token_index[c]] = 1.0
            dec_in[i, t + 1:, space_dec] = 1.0
            dec_out[i, t:, space_dec] = 1.0

        self.encoder_input  = torch.from_numpy(enc)
        self.decoder_input  = torch.from_numpy(dec_in)
        self.decoder_target = torch.from_numpy(dec_out)

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[idx], self.decoder_target[idx]


# ── Model ─────────────────────────────────────────────────────────────────

class Seq2SeqLSTM(nn.Module):
    """Character-level LSTM encoder–decoder with teacher forcing."""

    def __init__(
        self,
        num_encoder_tokens: int,
        num_decoder_tokens: int,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.encoder_lstm = nn.LSTM(num_encoder_tokens, latent_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(num_decoder_tokens, latent_dim, batch_first=True)
        self.dense = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        _, (h, c) = self.encoder_lstm(encoder_input)
        dec_out, _ = self.decoder_lstm(decoder_input, (h, c))
        return self.dense(dec_out)  # (B, T_dec, num_decoder_tokens)


# ── FL Interface ──────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return Seq2SeqLSTM(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size", config.get("batch_size", 64))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory", True)

    dataset_kwargs = config.get("dataset_kwargs", {})
    data_path = config.get("data_path", "fra.txt")
    full_dataset = Seq2SeqCharDataset(data_path=data_path, **dataset_kwargs)

    val_ratio = config.get("val_ratio", 0.2)
    n_val = max(1, int(len(full_dataset) * val_ratio))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
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
    ONE forward pass.  Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.
    """
    device = next(model.parameters()).device
    encoder_input, decoder_input, decoder_target = [b.to(device) for b in batch]

    logits = model(encoder_input, decoder_input)  # (B, T, V)

    B, T, V = logits.shape
    # decoder_target is one-hot; convert to class indices for CrossEntropyLoss
    targets = decoder_target.argmax(dim=-1).reshape(B * T)
    logits_flat = logits.reshape(B * T, V)

    loss = nn.CrossEntropyLoss()(logits_flat, targets)
    return loss
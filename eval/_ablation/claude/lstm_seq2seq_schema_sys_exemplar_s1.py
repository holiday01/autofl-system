"""
Auto-generated FL client module.
Original script: character-level recurrent sequence-to-sequence model (Keras → PyTorch).

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT:
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
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from pathlib import Path


# ── Model (converted from Keras to PyTorch) ──────────────────────────────────

class Seq2SeqModel(nn.Module):
    """
    Character-level encoder-decoder sequence-to-sequence model.
    Mirrors the Keras architecture in the original script:
      - Encoder LSTM consumes one-hot input sequences, yields (h, c) states.
      - Decoder LSTM is initialised with encoder states (teacher-forcing).
      - A linear projection maps decoder hidden states to target-vocab logits.
    CrossEntropyLoss (applied externally) is numerically equivalent to the
    original softmax + categorical_crossentropy combination.
    """

    def __init__(
        self,
        num_encoder_tokens: int = 71,
        num_decoder_tokens: int = 93,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_lstm = nn.LSTM(
            input_size=num_encoder_tokens,
            hidden_size=latent_dim,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=num_decoder_tokens,
            hidden_size=latent_dim,
            batch_first=True,
        )
        self.decoder_dense = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_input: (batch, enc_seq_len, num_encoder_tokens)  — one-hot
            decoder_input: (batch, dec_seq_len, num_decoder_tokens)  — one-hot
        Returns:
            logits:        (batch, dec_seq_len, num_decoder_tokens)
        """
        _, (state_h, state_c) = self.encoder_lstm(encoder_input)
        decoder_outputs, _ = self.decoder_lstm(decoder_input, (state_h, state_c))
        return self.decoder_dense(decoder_outputs)


# ── Dataset ───────────────────────────────────────────────────────────────────

class Seq2SeqCharDataset(Dataset):
    """
    Loads a tab-separated bilingual text file (e.g. fra.txt from the Anki dataset)
    and returns one-hot (encoder_input, decoder_input, decoder_target) triples.
    The preprocessing exactly mirrors the original Keras script.
    """

    def __init__(self, data_path: str, num_samples: int = 10000):
        input_texts, target_texts = [], []
        input_characters: set = set()
        target_characters: set = set()

        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        for line in lines[: min(num_samples, len(lines) - 1)]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            input_text = parts[0]
            target_text = "\t" + parts[1] + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                input_characters.add(char)
            for char in target_text:
                target_characters.add(char)

        input_characters_sorted = sorted(input_characters)
        target_characters_sorted = sorted(target_characters)

        self.num_encoder_tokens = len(input_characters_sorted)
        self.num_decoder_tokens = len(target_characters_sorted)
        self.max_encoder_seq_length = max(len(t) for t in input_texts)
        self.max_decoder_seq_length = max(len(t) for t in target_texts)

        input_token_index = {ch: i for i, ch in enumerate(input_characters_sorted)}
        target_token_index = {ch: i for i, ch in enumerate(target_characters_sorted)}

        n = len(input_texts)
        space_enc = input_token_index.get(" ", 0)
        space_dec = target_token_index.get(" ", 0)

        encoder_input_data = np.zeros(
            (n, self.max_encoder_seq_length, self.num_encoder_tokens), dtype=np.float32
        )
        decoder_input_data = np.zeros(
            (n, self.max_decoder_seq_length, self.num_decoder_tokens), dtype=np.float32
        )
        decoder_target_data = np.zeros(
            (n, self.max_decoder_seq_length, self.num_decoder_tokens), dtype=np.float32
        )

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            t_enc = 0
            for t_enc, char in enumerate(input_text):
                encoder_input_data[i, t_enc, input_token_index[char]] = 1.0
            if input_text:
                encoder_input_data[i, t_enc + 1:, space_enc] = 1.0

            t_dec = 0
            for t_dec, char in enumerate(target_text):
                decoder_input_data[i, t_dec, target_token_index[char]] = 1.0
                if t_dec > 0:
                    decoder_target_data[i, t_dec - 1, target_token_index[char]] = 1.0
            if target_text:
                decoder_input_data[i, t_dec + 1:, space_dec] = 1.0
                decoder_target_data[i, t_dec:, space_dec] = 1.0

        self.encoder_inputs = torch.from_numpy(encoder_input_data)
        self.decoder_inputs = torch.from_numpy(decoder_input_data)
        self.decoder_targets = torch.from_numpy(decoder_target_data)

    def __len__(self) -> int:
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        return (
            self.encoder_inputs[idx],
            self.decoder_inputs[idx],
            self.decoder_targets[idx],
        )


# ── FL Interface ──────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return Seq2SeqModel(**kwargs)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  16))
    num_workers = local.get("num_workers", config.get("num_workers", 2))
    pin_memory  = local.get("pin_memory",  True)
    num_samples = config.get("num_samples", 10000)

    data_path = config.get("data_path", ".")

    # Accept either a direct path to fra.txt or a directory containing it
    if os.path.isfile(data_path):
        txt_path = data_path
    else:
        txt_path = os.path.join(data_path, "fra.txt")

    if not os.path.isfile(txt_path):
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Translation data file not found at '{txt_path}'. "
                "Set config['data_path'] to the directory or file containing 'fra.txt', "
                "or set config['allow_synthetic_data'] = True to use synthetic data for testing."
            )
        # Synthetic fallback — gated on allow_synthetic_data
        model_kwargs       = config.get("model_kwargs", {})
        num_enc_tokens     = model_kwargs.get("num_encoder_tokens", 71)
        num_dec_tokens     = model_kwargs.get("num_decoder_tokens", 93)
        max_enc_len        = config.get("max_encoder_seq_length", 16)
        max_dec_len        = config.get("max_decoder_seq_length", 59)
        n_synth            = 200

        enc_in = torch.zeros(n_synth, max_enc_len, num_enc_tokens)
        enc_in.scatter_(2, torch.randint(0, num_enc_tokens, (n_synth, max_enc_len, 1)), 1.0)

        dec_in = torch.zeros(n_synth, max_dec_len, num_dec_tokens)
        dec_in.scatter_(2, torch.randint(0, num_dec_tokens, (n_synth, max_dec_len, 1)), 1.0)

        dec_tgt = torch.zeros(n_synth, max_dec_len, num_dec_tokens)
        dec_tgt.scatter_(2, torch.randint(0, num_dec_tokens, (n_synth, max_dec_len, 1)), 1.0)

        full_dataset = TensorDataset(enc_in, dec_in, dec_tgt)
    else:
        full_dataset = Seq2SeqCharDataset(data_path=txt_path, num_samples=num_samples)

    val_ratio = config.get("val_ratio", 0.1)
    n_val     = max(1, int(len(full_dataset) * val_ratio))
    n_train   = len(full_dataset) - n_val
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

    Batch layout (list/tuple):
        batch[0]: encoder_input  (batch, enc_seq_len, num_encoder_tokens) — one-hot
        batch[1]: decoder_input  (batch, dec_seq_len, num_decoder_tokens) — one-hot
        batch[2]: decoder_target (batch, dec_seq_len, num_decoder_tokens) — one-hot

    Batch layout (dict) — keys: "encoder_input", "decoder_input", "decoder_target".
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        encoder_input  = batch[0]
        decoder_input  = batch[1]
        decoder_target = batch[2]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        encoder_input  = batch["encoder_input"]
        decoder_input  = batch["decoder_input"]
        decoder_target = batch["decoder_target"]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # Forward pass — returns (batch, dec_seq_len, num_decoder_tokens) logits
    logits = model(encoder_input, decoder_input)

    # Convert one-hot decoder_target to class indices required by CrossEntropyLoss
    # decoder_target shape: (batch, dec_seq_len, num_decoder_tokens)
    target_indices = decoder_target.argmax(dim=-1)  # (batch, dec_seq_len)

    # CrossEntropyLoss expects logits as (N, C, d...) and targets as (N, d...)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(
        logits.permute(0, 2, 1),  # (batch, num_decoder_tokens, dec_seq_len)
        target_indices,            # (batch, dec_seq_len)
    )
    return loss
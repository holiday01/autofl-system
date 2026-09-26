"""
Auto-generated FL client module.
Original script: character-level recurrent sequence-to-sequence model (Keras → PyTorch)

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
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


def _download_fra_eng(cache_dir: str = None) -> str:
    url = "http://www.manythings.org/anki/fra-eng.zip"
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fra_eng")
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "fra-eng.zip")
    txt_path = os.path.join(cache_dir, "fra.txt")
    if not os.path.isfile(txt_path):
        if not os.path.isfile(zip_path):
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("fra.txt", cache_dir)
    return txt_path


class Seq2SeqDataset(Dataset):
    """Character-level one-hot encoded English→French translation pairs."""

    def __init__(self, data_path: str = None, num_samples: int = 10000, download: bool = True):
        if data_path is None and download:
            data_path = _download_fra_eng()

        if data_path and os.path.isfile(data_path):
            input_texts, target_texts = [], []
            input_chars, target_chars = set(), set()
            with open(data_path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
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
            self.num_encoder_tokens = len(input_chars)
            self.num_decoder_tokens = len(target_chars)
            max_enc = max(len(t) for t in input_texts)
            max_dec = max(len(t) for t in target_texts)
            inp_idx = {c: i for i, c in enumerate(input_chars)}
            tgt_idx = {c: i for i, c in enumerate(target_chars)}
            space_enc = inp_idx.get(" ", 0)
            space_dec = tgt_idx.get(" ", 0)

            n = len(input_texts)
            enc_data = np.zeros((n, max_enc, self.num_encoder_tokens), dtype=np.float32)
            dec_in   = np.zeros((n, max_dec, self.num_decoder_tokens), dtype=np.float32)
            dec_out  = np.zeros((n, max_dec, self.num_decoder_tokens), dtype=np.float32)

            for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
                t = -1
                for t, char in enumerate(inp):
                    enc_data[i, t, inp_idx[char]] = 1.0
                if t >= 0:
                    enc_data[i, t + 1:, space_enc] = 1.0
                t = -1
                for t, char in enumerate(tgt):
                    dec_in[i, t, tgt_idx[char]] = 1.0
                    if t > 0:
                        dec_out[i, t - 1, tgt_idx[char]] = 1.0
                if t >= 0:
                    dec_in[i, t + 1:, space_dec] = 1.0
                    dec_out[i, t:, space_dec] = 1.0

            self.enc = enc_data
            self.dec_in = dec_in
            self.dec_out = dec_out
        else:
            # synthetic fallback for testing without data
            self.num_encoder_tokens = 71
            self.num_decoder_tokens = 93
            n, enc_len, dec_len = 200, 16, 20
            self.enc    = np.eye(self.num_encoder_tokens, dtype=np.float32)[
                np.random.randint(0, self.num_encoder_tokens, (n, enc_len))]
            self.dec_in = np.eye(self.num_decoder_tokens, dtype=np.float32)[
                np.random.randint(0, self.num_decoder_tokens, (n, dec_len))]
            self.dec_out = np.eye(self.num_decoder_tokens, dtype=np.float32)[
                np.random.randint(0, self.num_decoder_tokens, (n, dec_len))]

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.enc[idx]),
            torch.from_numpy(self.dec_in[idx]),
            torch.from_numpy(self.dec_out[idx]),
        )


class Seq2SeqLSTM(nn.Module):
    def __init__(self, num_encoder_tokens: int, num_decoder_tokens: int, latent_dim: int = 256):
        super().__init__()
        self.encoder_lstm = nn.LSTM(num_encoder_tokens, latent_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(num_decoder_tokens, latent_dim, batch_first=True)
        self.output_proj  = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder_lstm(encoder_input)
        dec_out, _ = self.decoder_lstm(decoder_input, (h, c))
        return self.output_proj(dec_out)  # (batch, dec_seq_len, num_decoder_tokens)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    num_encoder_tokens = kwargs.get("num_encoder_tokens", config.get("num_encoder_tokens", 71))
    num_decoder_tokens = kwargs.get("num_decoder_tokens", config.get("num_decoder_tokens", 93))
    latent_dim         = kwargs.get("latent_dim",         config.get("latent_dim",         256))
    return Seq2SeqLSTM(
        num_encoder_tokens=num_encoder_tokens,
        num_decoder_tokens=num_decoder_tokens,
        latent_dim=latent_dim,
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  64))
    num_workers = local.get("num_workers", config.get("num_workers",  2))
    pin_memory  = local.get("pin_memory",  True)

    dataset_kwargs = config.get("dataset_kwargs", {})
    data_path      = config.get("data_path", None)
    full_dataset   = Seq2SeqDataset(data_path=data_path, **dataset_kwargs)

    val_ratio = config.get("val_ratio", 0.2)
    n_val     = max(1, int(len(full_dataset) * val_ratio))
    n_train   = len(full_dataset) - n_val
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
    ONE forward pass. Returns the raw loss tensor WITH grad_fn attached.
    The FL runtime calls loss.backward() and optimizer.step() externally —
    do NOT do either here, and do NOT detach() or .item() the returned loss.

    Expected batch layout: (encoder_input, decoder_input, decoder_target)
      encoder_input  : (B, enc_seq_len, num_encoder_tokens)  one-hot float32
      decoder_input  : (B, dec_seq_len, num_decoder_tokens)  one-hot float32 (teacher-forced)
      decoder_target : (B, dec_seq_len, num_decoder_tokens)  one-hot float32 (shifted by 1)
    """
    device = next(model.parameters()).device

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        encoder_input, decoder_input, decoder_target = batch[0], batch[1], batch[2]
    elif isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        encoder_input  = batch.get("encoder_input",  batch.get("x"))
        decoder_input  = batch.get("decoder_input")
        decoder_target = batch.get("decoder_target", batch.get("y"))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    logits = model(encoder_input, decoder_input)  # (B, dec_seq_len, num_decoder_tokens)

    # one-hot → class indices for CrossEntropyLoss
    target_indices = decoder_target.argmax(dim=-1)  # (B, dec_seq_len)
    B, S, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * S, V), target_indices.reshape(B * S))
    return loss
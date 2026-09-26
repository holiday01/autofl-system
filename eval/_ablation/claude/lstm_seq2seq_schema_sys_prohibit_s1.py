import numpy as np
import keras
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# ─── Model (Keras/TF → PyTorch) ──────────────────────────────────────────────

class Seq2SeqModel(nn.Module):
    """Character-level encoder-decoder seq2seq model (LSTM-based).

    Mirrors the original Keras architecture:
      encoder LSTM  →  (h, c)  →  decoder LSTM  →  linear projection
    """

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
        # encoder_input : (B, enc_len, num_encoder_tokens)
        # decoder_input : (B, dec_len, num_decoder_tokens)
        _, (h, c) = self.encoder_lstm(encoder_input)
        decoder_output, _ = self.decoder_lstm(decoder_input, (h, c))
        logits = self.decoder_dense(decoder_output)   # (B, dec_len, num_decoder_tokens)
        return logits


# ─── Internal helper ─────────────────────────────────────────────────────────

def _build_dataset_from_file(
    fra_txt: str,
    num_samples: int,
    num_encoder_tokens: int,
    num_decoder_tokens: int,
) -> TensorDataset:
    """Vectorise fra.txt into one-hot float tensors identical to the original script."""
    input_texts: list = []
    target_texts: list = []
    input_characters: set = set()
    target_characters: set = set()

    with open(fra_txt, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    for line in lines[: min(num_samples, len(lines) - 1)]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        input_text, target_text = parts[0], parts[1]
        target_text = "\t" + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            input_characters.add(char)
        for char in target_text:
            target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))

    # Allocate at least as many token slots as the model expects
    _num_enc = max(len(input_characters), num_encoder_tokens)
    _num_dec = max(len(target_characters), num_decoder_tokens)

    max_encoder_seq_length = max(len(t) for t in input_texts)
    max_decoder_seq_length = max(len(t) for t in target_texts)

    input_token_index  = {char: i for i, char in enumerate(input_characters)}
    target_token_index = {char: i for i, char in enumerate(target_characters)}

    n = len(input_texts)
    encoder_input_data  = np.zeros((n, max_encoder_seq_length, _num_enc), dtype="float32")
    decoder_input_data  = np.zeros((n, max_decoder_seq_length, _num_dec), dtype="float32")
    decoder_target_data = np.zeros((n, max_decoder_seq_length, _num_dec), dtype="float32")

    space_enc_idx = input_token_index.get(" ", 0)
    space_dec_idx = target_token_index.get(" ", 0)

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        t_enc = 0
        for t_enc, char in enumerate(input_text):
            encoder_input_data[i, t_enc, input_token_index[char]] = 1.0
        encoder_input_data[i, t_enc + 1 :, space_enc_idx] = 1.0

        t_dec = 0
        for t_dec, char in enumerate(target_text):
            decoder_input_data[i, t_dec, target_token_index[char]] = 1.0
            if t_dec > 0:
                decoder_target_data[i, t_dec - 1, target_token_index[char]] = 1.0
        decoder_input_data[i,  t_dec + 1 :, space_dec_idx] = 1.0
        decoder_target_data[i, t_dec :,     space_dec_idx] = 1.0

    return TensorDataset(
        torch.from_numpy(encoder_input_data),
        torch.from_numpy(decoder_input_data),
        torch.from_numpy(decoder_target_data),
    )


# ─── FL API ──────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate the seq2seq model from config."""
    kwargs = config.get("model_kwargs", {})
    model = Seq2SeqModel(
        num_encoder_tokens=kwargs.get("num_encoder_tokens", 71),
        num_decoder_tokens=kwargs.get("num_decoder_tokens", 93),
        latent_dim=kwargs.get("latent_dim", 256),
    )
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Data source priority:
      1. config['data_path'] / fra.txt
      2. Keras cache (automatic download + unzip)
      3. Synthetic tensors – ONLY when config['allow_synthetic_data'] is True;
         raises FileNotFoundError otherwise.
    """
    local_cfg          = config.get("local", {})
    batch_size         = local_cfg.get("batch_size", 16)
    data_path          = config.get("data_path", ".")
    num_samples        = config.get("num_samples", 10000)
    val_fraction       = config.get("val_fraction", 0.2)

    model_kwargs       = config.get("model_kwargs", {})
    num_encoder_tokens = model_kwargs.get("num_encoder_tokens", 71)
    num_decoder_tokens = model_kwargs.get("num_decoder_tokens", 93)

    # ── Locate fra.txt ───────────────────────────────────────────────────────
    fra_txt = os.path.join(data_path, "fra.txt")
    if not os.path.isfile(fra_txt):
        try:
            fpath   = keras.utils.get_file(origin="http://www.manythings.org/anki/fra-eng.zip")
            dirpath = Path(fpath).parent.absolute()
            os.system(f"unzip -q {fpath} -d {dirpath}")
            fra_txt = os.path.join(str(dirpath), "fra.txt")
        except Exception:
            fra_txt = None

    # ── Build dataset ────────────────────────────────────────────────────────
    if fra_txt is None or not os.path.isfile(fra_txt):
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"fra.txt not found under data_path='{data_path}' and automatic download failed. "
                "Set config['allow_synthetic_data'] = True to use random tensors for smoke-testing."
            )
        # Synthetic fallback ─ gated on allow_synthetic_data
        n_synth    = min(num_samples, 200)
        max_enc    = config.get("max_encoder_seq_length", 20)
        max_dec    = config.get("max_decoder_seq_length", 25)
        enc        = torch.randn(n_synth, max_enc, num_encoder_tokens)
        dec_in     = torch.randn(n_synth, max_dec, num_decoder_tokens)
        idx        = torch.randint(0, num_decoder_tokens, (n_synth, max_dec))
        dec_tgt    = torch.zeros(n_synth, max_dec, num_decoder_tokens)
        dec_tgt.scatter_(2, idx.unsqueeze(-1), 1.0)
        dataset    = TensorDataset(enc, dec_in, dec_tgt)
    else:
        dataset = _build_dataset_from_file(
            fra_txt, num_samples, num_encoder_tokens, num_decoder_tokens
        )

    # ── Train / val split ────────────────────────────────────────────────────
    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_set if split == "train" else val_set
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=False,
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """Single forward pass; returns the loss tensor (grad_fn intact).

    The FL runtime is responsible for loss.backward() and optimizer.step().
    """
    device = next(model.parameters()).device

    enc_input, dec_input, dec_target = batch
    enc_input  = enc_input.to(device)
    dec_input  = dec_input.to(device)
    dec_target = dec_target.to(device)

    logits = model(enc_input, dec_input)   # (B, T, num_decoder_tokens)

    B, T, C   = logits.shape
    logits_2d = logits.reshape(B * T, C)

    # Convert one-hot dec_target to class indices for cross_entropy
    target_indices = dec_target.reshape(B * T, C).argmax(dim=-1)   # (B*T,)

    loss = F.cross_entropy(logits_2d, target_indices)
    return loss
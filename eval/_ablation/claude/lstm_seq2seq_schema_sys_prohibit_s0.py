import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class Seq2SeqCharModel(nn.Module):
    """
    PyTorch equivalent of the Keras character-level seq2seq model.

    Encoder:  single-layer LSTM – only final (h, c) states are kept.
    Decoder:  single-layer LSTM initialised with encoder states,
              followed by a linear projection to decoder-token logits.
    """

    def __init__(
        self,
        num_encoder_tokens: int = 71,
        num_decoder_tokens: int = 93,
        latent_dim: int = 256,
    ):
        super().__init__()
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
        # No softmax here — F.cross_entropy expects raw logits
        self.decoder_dense = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(
        self,
        encoder_input: torch.Tensor,   # (B, T_enc, num_encoder_tokens)
        decoder_input: torch.Tensor,   # (B, T_dec, num_decoder_tokens)
    ) -> torch.Tensor:                 # (B, T_dec, num_decoder_tokens) logits
        _, (state_h, state_c) = self.encoder_lstm(encoder_input)
        decoder_output, _ = self.decoder_lstm(decoder_input, (state_h, state_c))
        logits = self.decoder_dense(decoder_output)
        return logits


# ---------------------------------------------------------------------------
# Internal data-loading helper
# ---------------------------------------------------------------------------

def _load_seq2seq_data(data_path: str, num_samples: int = 10000):
    """
    Vectorise fra.txt into one-hot NumPy arrays exactly as the original script.

    Returns
    -------
    encoder_input_data  : float32 ndarray  (N, max_enc_len, num_enc_tok)
    decoder_input_data  : float32 ndarray  (N, max_dec_len, num_dec_tok)
    decoder_target_data : float32 ndarray  (N, max_dec_len, num_dec_tok)
    num_encoder_tokens  : int
    num_decoder_tokens  : int
    """
    fra_path = os.path.join(data_path, "fra.txt")
    if not os.path.isfile(fra_path):
        raise FileNotFoundError(
            f"Expected 'fra.txt' at '{fra_path}'. "
            "Download http://www.manythings.org/anki/fra-eng.zip and extract it "
            "into the directory pointed to by config['data_path']."
        )

    input_texts: list[str] = []
    target_texts: list[str] = []
    input_characters: set[str] = set()
    target_characters: set[str] = set()

    with open(fra_path, "r", encoding="utf-8") as f:
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

    input_characters = sorted(input_characters)
    target_characters = sorted(target_characters)
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max(len(txt) for txt in input_texts)
    max_decoder_seq_length = max(len(txt) for txt in target_texts)

    input_token_index = {char: i for i, char in enumerate(input_characters)}
    target_token_index = {char: i for i, char in enumerate(target_characters)}

    # Padding fallback indices (space is virtually always present)
    enc_pad_idx = input_token_index.get(" ", 0)
    dec_pad_idx = target_token_index.get(" ", 0)

    n = len(input_texts)
    encoder_input_data = np.zeros(
        (n, max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (n, max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (n, max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        # --- encoder ---
        t_enc = 0
        for t_enc, char in enumerate(input_text):
            encoder_input_data[i, t_enc, input_token_index[char]] = 1.0
        encoder_input_data[i, t_enc + 1 :, enc_pad_idx] = 1.0

        # --- decoder (teacher-forcing offset) ---
        t_dec = 0
        for t_dec, char in enumerate(target_text):
            decoder_input_data[i, t_dec, target_token_index[char]] = 1.0
            if t_dec > 0:
                decoder_target_data[i, t_dec - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t_dec + 1 :, dec_pad_idx] = 1.0
        decoder_target_data[i, t_dec :, dec_pad_idx] = 1.0

    return (
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        num_encoder_tokens,
        num_decoder_tokens,
    )


# ---------------------------------------------------------------------------
# FL API
# ---------------------------------------------------------------------------

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the Seq2SeqCharModel.

    Recognised config keys (all optional):
        model_kwargs.num_encoder_tokens  (default 71)
        model_kwargs.num_decoder_tokens  (default 93)
        model_kwargs.latent_dim          (default 256)
    """
    kwargs = config.get("model_kwargs", {})
    return Seq2SeqCharModel(
        num_encoder_tokens=kwargs.get("num_encoder_tokens", 71),
        num_decoder_tokens=kwargs.get("num_decoder_tokens", 93),
        latent_dim=kwargs.get("latent_dim", 256),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ('train' or 'val').

    Each batch is a 3-tuple:
        (encoder_input, decoder_input, decoder_target)
    all as float32 tensors with one-hot encoding on the last axis.

    Recognised config keys:
        data_path              – directory containing fra.txt  (default '.')
        local.batch_size       – mini-batch size               (default 16)
        num_samples            – rows to read from fra.txt     (default 10000)
        allow_synthetic_data   – enable synthetic fallback     (default False)
        model_kwargs           – forwarded to synthetic-data sizing
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    num_samples = config.get("num_samples", 10000)
    allow_synthetic = config.get("allow_synthetic_data", False)

    try:
        enc_in, dec_in, dec_tgt, _, _ = _load_seq2seq_data(data_path, num_samples)
        dataset = TensorDataset(
            torch.tensor(enc_in,  dtype=torch.float32),
            torch.tensor(dec_in,  dtype=torch.float32),
            torch.tensor(dec_tgt, dtype=torch.float32),
        )

    except FileNotFoundError as exc:
        if not allow_synthetic:
            raise FileNotFoundError(
                "Real dataset unavailable and config['allow_synthetic_data'] is False. "
                f"Details: {exc}"
            ) from exc

        # ------------------------------------------------------------------ #
        # Synthetic fallback – only reached when allow_synthetic_data is True #
        # ------------------------------------------------------------------ #
        model_kwargs = config.get("model_kwargs", {})
        num_enc_tok = model_kwargs.get("num_encoder_tokens", 71)
        num_dec_tok = model_kwargs.get("num_decoder_tokens", 93)
        n_syn, enc_seq_len, dec_seq_len = 200, 16, 20

        enc_syn = torch.zeros(n_syn, enc_seq_len, num_enc_tok)
        enc_syn.scatter_(2, torch.randint(0, num_enc_tok, (n_syn, enc_seq_len, 1)), 1.0)

        dec_in_syn = torch.zeros(n_syn, dec_seq_len, num_dec_tok)
        dec_in_syn.scatter_(2, torch.randint(0, num_dec_tok, (n_syn, dec_seq_len, 1)), 1.0)

        dec_tgt_syn = torch.zeros(n_syn, dec_seq_len, num_dec_tok)
        dec_tgt_syn.scatter_(2, torch.randint(0, num_dec_tok, (n_syn, dec_seq_len, 1)), 1.0)

        dataset = TensorDataset(enc_syn, dec_in_syn, dec_tgt_syn)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    chosen = train_ds if split == "train" else val_ds
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(
    model: nn.Module,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Execute one forward pass and return the loss tensor.

    The FL runtime is responsible for loss.backward() and optimizer.step().
    This function must NOT call either.
    """
    enc_input, dec_input, dec_target = batch

    device = next(model.parameters()).device
    enc_input  = enc_input.to(device)
    dec_input  = dec_input.to(device)
    dec_target = dec_target.to(device)

    # Forward pass
    logits = model(enc_input, dec_input)   # (B, T, num_decoder_tokens)

    # Reshape for F.cross_entropy: (B*T, C) and (B*T,) class indices
    B, T, C = logits.shape
    logits_flat   = logits.reshape(B * T, C)
    target_indices = dec_target.reshape(B * T, C).argmax(dim=-1)

    loss = F.cross_entropy(logits_flat, target_indices)
    return loss  # grad_fn intact — do NOT detach or call .item()
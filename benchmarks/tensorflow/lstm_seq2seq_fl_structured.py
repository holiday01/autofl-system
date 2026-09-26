import numpy as np
import os
from pathlib import Path

try:
    import keras
except ImportError:
    keras = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# ── Vocabulary / shape defaults (used when real data is unavailable) ──────────
_DEFAULT_NUM_ENCODER_TOKENS = 71
_DEFAULT_NUM_DECODER_TOKENS = 93
_DEFAULT_MAX_ENCODER_SEQ_LEN = 15
_DEFAULT_MAX_DECODER_SEQ_LEN = 59
_DEFAULT_LATENT_DIM = 256
_DEFAULT_NUM_SAMPLES = 10000


# ── Model ─────────────────────────────────────────────────────────────────────

class Seq2SeqModel(nn.Module):
    """Character-level encoder-decoder seq2seq model.

    Direct PyTorch equivalent of the original Keras architecture:
      - Encoder LSTM  → final (h, c) hidden states
      - Decoder LSTM  → full output sequence (teacher-forcing)
      - Linear + softmax projection to target vocabulary
    """

    def __init__(
        self,
        num_encoder_tokens: int = _DEFAULT_NUM_ENCODER_TOKENS,
        num_decoder_tokens: int = _DEFAULT_NUM_DECODER_TOKENS,
        latent_dim: int = _DEFAULT_LATENT_DIM,
    ):
        super().__init__()
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.latent_dim = latent_dim

        # Encoder: one LSTM layer, we only keep the final state.
        self.encoder_lstm = nn.LSTM(
            input_size=num_encoder_tokens,
            hidden_size=latent_dim,
            batch_first=True,
        )
        # Decoder: one LSTM layer initialised with encoder states,
        # returns the full output sequence.
        self.decoder_lstm = nn.LSTM(
            input_size=num_decoder_tokens,
            hidden_size=latent_dim,
            batch_first=True,
        )
        # Dense projection to target vocabulary (logits; softmax applied in loss).
        self.decoder_dense = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_input: (batch, enc_seq_len, num_encoder_tokens)  one-hot
            decoder_input: (batch, dec_seq_len, num_decoder_tokens)  one-hot

        Returns:
            logits: (batch, dec_seq_len, num_decoder_tokens)
        """
        # Encode – discard output, keep final hidden state.
        _, (state_h, state_c) = self.encoder_lstm(encoder_input)
        # state_h / state_c: (1, batch, latent_dim)

        # Decode with teacher forcing, seeded by encoder state.
        decoder_out, _ = self.decoder_lstm(decoder_input, (state_h, state_c))
        # decoder_out: (batch, dec_seq_len, latent_dim)

        logits = self.decoder_dense(decoder_out)
        # logits: (batch, dec_seq_len, num_decoder_tokens)
        return logits


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _load_real_data(fra_txt_path: str, num_samples: int = _DEFAULT_NUM_SAMPLES):
    """Parse fra.txt, build one-hot arrays identical to the original script.

    Returns
    -------
    encoder_input_data  : np.ndarray  (N, max_enc_len, num_enc_tok)
    decoder_input_data  : np.ndarray  (N, max_dec_len, num_dec_tok)
    decoder_target_data : np.ndarray  (N, max_dec_len, num_dec_tok)
    num_encoder_tokens  : int
    num_decoder_tokens  : int
    """
    with open(fra_txt_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    input_texts: list[str] = []
    target_texts: list[str] = []
    input_characters: set[str] = set()
    target_characters: set[str] = set()

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

    input_characters_sorted = sorted(input_characters)
    target_characters_sorted = sorted(target_characters)
    num_encoder_tokens = len(input_characters_sorted)
    num_decoder_tokens = len(target_characters_sorted)
    max_encoder_seq_length = max(len(t) for t in input_texts)
    max_decoder_seq_length = max(len(t) for t in target_texts)

    input_token_index = {c: i for i, c in enumerate(input_characters_sorted)}
    target_token_index = {c: i for i, c in enumerate(target_characters_sorted)}

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

    for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
        t = 0
        for t, char in enumerate(inp):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, input_token_index.get(" ", 0)] = 1.0
        t = 0
        for t, char in enumerate(tgt):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index.get(" ", 0)] = 1.0
        decoder_target_data[i, t:, target_token_index.get(" ", 0)] = 1.0

    return (
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        num_encoder_tokens,
        num_decoder_tokens,
    )


class _Seq2SeqDataset(Dataset):
    """Wraps pre-built one-hot numpy arrays as a PyTorch Dataset."""

    def __init__(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
        decoder_targets: torch.Tensor,
    ):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_targets = decoder_targets

    def __len__(self) -> int:
        return self.encoder_inputs.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.encoder_inputs[idx],
            self.decoder_inputs[idx],
            self.decoder_targets[idx],
        )


def _make_synthetic_dataset(
    n: int = 200,
    enc_len: int = _DEFAULT_MAX_ENCODER_SEQ_LEN,
    dec_len: int = _DEFAULT_MAX_DECODER_SEQ_LEN,
    enc_tok: int = _DEFAULT_NUM_ENCODER_TOKENS,
    dec_tok: int = _DEFAULT_NUM_DECODER_TOKENS,
) -> _Seq2SeqDataset:
    """Return a synthetic one-hot dataset for when real data is unavailable."""
    enc_in = torch.zeros(n, enc_len, enc_tok)
    enc_in.scatter_(2, torch.randint(0, enc_tok, (n, enc_len, 1)), 1.0)

    dec_in = torch.zeros(n, dec_len, dec_tok)
    dec_in.scatter_(2, torch.randint(0, dec_tok, (n, dec_len, 1)), 1.0)

    dec_tgt = torch.zeros(n, dec_len, dec_tok)
    dec_tgt.scatter_(2, torch.randint(0, dec_tok, (n, dec_len, 1)), 1.0)

    return _Seq2SeqDataset(enc_in, dec_in, dec_tgt)


# ── FL Public API ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the Seq2SeqModel.

    Relevant config keys (all under ``model_kwargs``):
      - ``num_encoder_tokens``  (int, default 71)
      - ``num_decoder_tokens``  (int, default 93)
      - ``latent_dim``          (int, default 256)
    """
    kwargs = config.get("model_kwargs", {})
    return Seq2SeqModel(
        num_encoder_tokens=kwargs.get("num_encoder_tokens", _DEFAULT_NUM_ENCODER_TOKENS),
        num_decoder_tokens=kwargs.get("num_decoder_tokens", _DEFAULT_NUM_DECODER_TOKENS),
        latent_dim=kwargs.get("latent_dim", _DEFAULT_LATENT_DIM),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for the requested split.

    Config keys:
      - ``data_path``                  : directory that contains ``fra.txt``
      - ``local.batch_size``           : mini-batch size (default 16)
      - ``num_samples``                : max lines to load (default 10 000)

    Falls back to synthetic one-hot data when ``fra.txt`` cannot be read.
    """
    batch_size: int = config.get("local", {}).get("batch_size", 16)
    data_path: str = config.get("data_path", ".")
    num_samples: int = config.get("num_samples", _DEFAULT_NUM_SAMPLES)

    fra_txt = os.path.join(data_path, "fra.txt")

    try:
        enc_in_np, dec_in_np, dec_tgt_np, _, _ = _load_real_data(fra_txt, num_samples)
        dataset: Dataset = _Seq2SeqDataset(
            torch.from_numpy(enc_in_np),
            torch.from_numpy(dec_in_np),
            torch.from_numpy(dec_tgt_np),
        )
    except Exception:
        # Real data unavailable – use synthetic tensors so FL training can proceed.
        dataset = _make_synthetic_dataset()

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    chosen = train_ds if split == "train" else val_ds

    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=False,
    )


def train_step(
    model: nn.Module,
    batch,
    optimizer,  # noqa: ARG001 – held by the FL runtime; not used here
    config: dict,  # noqa: ARG001
) -> torch.Tensor:
    """Run ONE forward pass and return the loss tensor with grad attached.

    The FL runtime is responsible for calling ``loss.backward()`` and
    ``optimizer.step()``.  This function must NOT do either.

    Each element of ``batch``:
      - encoder_input   : (batch, enc_seq_len, num_encoder_tokens)  float32
      - decoder_input   : (batch, dec_seq_len, num_decoder_tokens)  float32
      - decoder_target  : (batch, dec_seq_len, num_decoder_tokens)  float32  one-hot
    """
    device = next(model.parameters()).device

    encoder_input, decoder_input, decoder_target = batch
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    decoder_target = decoder_target.to(device)

    # Forward pass (teacher forcing handled by the model).
    logits = model(encoder_input, decoder_input)
    # logits: (batch, dec_seq_len, num_decoder_tokens)

    # Convert one-hot targets → class indices for CrossEntropyLoss.
    # This mirrors Keras's categorical_crossentropy.
    target_indices = decoder_target.argmax(dim=-1)  # (batch, dec_seq_len)

    # CrossEntropyLoss expects channel dim second: (batch, C, seq_len).
    loss = F.cross_entropy(
        logits.permute(0, 2, 1),  # (batch, num_decoder_tokens, dec_seq_len)
        target_indices,           # (batch, dec_seq_len)
    )
    # loss has grad_fn – returned as-is for the FL runtime to differentiate.
    return loss
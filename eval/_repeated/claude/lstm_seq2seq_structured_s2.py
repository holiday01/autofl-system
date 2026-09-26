import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ── Model (PyTorch port of the Keras encoder-decoder LSTM) ────────────────────

class Seq2SeqModel(nn.Module):
    """Character-level sequence-to-sequence model.

    Encoder:  LSTM  → final (h, c) states
    Decoder:  LSTM  initialised with encoder states → full sequence output
    Projection: Linear + Softmax  → per-step character probabilities
    """

    def __init__(
        self,
        num_encoder_tokens: int,
        num_decoder_tokens: int,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            num_encoder_tokens, latent_dim, batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            num_decoder_tokens, latent_dim, batch_first=True
        )
        self.decoder_dense = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(
        self,
        encoder_input: torch.Tensor,   # (B, T_enc, num_encoder_tokens)
        decoder_input: torch.Tensor,   # (B, T_dec, num_decoder_tokens)
    ) -> torch.Tensor:                 # (B, T_dec, num_decoder_tokens)
        # Encode: discard per-step outputs, keep final states
        _, (h, c) = self.encoder_lstm(encoder_input)
        # Decode: full sequence, seeded with encoder states
        decoder_out, _ = self.decoder_lstm(decoder_input, (h, c))
        # Project to vocabulary and normalise
        logits = self.decoder_dense(decoder_out)
        return F.softmax(logits, dim=-1)


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _build_arrays(data_path: str, num_samples: int = 10000):
    """Parse fra.txt and return one-hot numpy arrays + vocab sizes."""
    input_texts: list[str] = []
    target_texts: list[str] = []
    input_characters: set[str] = set()
    target_characters: set[str] = set()

    with open(data_path, "r", encoding="utf-8") as fh:
        lines = fh.read().split("\n")

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

    input_characters_list  = sorted(input_characters)
    target_characters_list = sorted(target_characters)
    num_encoder_tokens     = len(input_characters_list)
    num_decoder_tokens     = len(target_characters_list)
    max_enc_len = max(len(t) for t in input_texts)
    max_dec_len = max(len(t) for t in target_texts)

    input_token_index  = {ch: i for i, ch in enumerate(input_characters_list)}
    target_token_index = {ch: i for i, ch in enumerate(target_characters_list)}

    # Fall-back indices for padding with space (mirrors the original script)
    enc_pad_idx = input_token_index.get(" ", 0)
    dec_pad_idx = target_token_index.get(" ", 0)

    n = len(input_texts)
    encoder_input_data  = np.zeros((n, max_enc_len, num_encoder_tokens),  dtype="float32")
    decoder_input_data  = np.zeros((n, max_dec_len, num_decoder_tokens),  dtype="float32")
    decoder_target_data = np.zeros((n, max_dec_len, num_decoder_tokens),  dtype="float32")

    for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
        t = 0
        for t, ch in enumerate(inp):
            encoder_input_data[i, t, input_token_index[ch]] = 1.0
        encoder_input_data[i, t + 1:, enc_pad_idx] = 1.0

        t = 0
        for t, ch in enumerate(tgt):
            decoder_input_data[i, t, target_token_index[ch]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[ch]] = 1.0
        decoder_input_data[i,  t + 1:, dec_pad_idx] = 1.0
        decoder_target_data[i, t:,     dec_pad_idx] = 1.0

    return (
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        num_encoder_tokens,
        num_decoder_tokens,
    )


class Seq2SeqDataset(Dataset):
    """Wraps (encoder_input, decoder_input, decoder_target) numpy arrays."""

    def __init__(
        self,
        encoder_input:  np.ndarray,
        decoder_input:  np.ndarray,
        decoder_target: np.ndarray,
    ):
        self.enc = torch.from_numpy(encoder_input)
        self.dec = torch.from_numpy(decoder_input)
        self.tgt = torch.from_numpy(decoder_target)

    def __len__(self) -> int:
        return self.enc.shape[0]

    def __getitem__(self, idx):
        return self.enc[idx], self.dec[idx], self.tgt[idx]


# ── FL API ─────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Instantiate and return the Seq2SeqModel.

    Expected config keys (all under 'model_kwargs'):
        num_encoder_tokens  (int, default 71)  – encoder one-hot width
        num_decoder_tokens  (int, default 93)  – decoder one-hot width
        latent_dim          (int, default 256) – LSTM hidden size
    """
    kwargs = config.get("model_kwargs", {})
    # Defaults match the fra-eng character vocabulary sizes from the original script
    num_encoder_tokens = int(kwargs.get("num_encoder_tokens", 71))
    num_decoder_tokens = int(kwargs.get("num_decoder_tokens", 93))
    latent_dim         = int(kwargs.get("latent_dim",          256))
    return Seq2SeqModel(num_encoder_tokens, num_decoder_tokens, latent_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Return a DataLoader for *split* ('train' or 'val').

    config keys
    -----------
    data_path               : directory that contains fra.txt
    local.batch_size        : batch size (default 16)
    num_samples             : max sentences to load (default 10000)
    allow_synthetic_data    : must be True to permit the random-tensor fallback
    model_kwargs.*          : forwarded to synthetic tensor shape when real data absent
    """
    local_cfg   = config.get("local", {})
    batch_size  = int(local_cfg.get("batch_size", 16))
    data_path   = config.get("data_path", ".")
    num_samples = int(config.get("num_samples", 10000))

    fra_txt = os.path.join(data_path, "fra.txt")

    if os.path.isfile(fra_txt):
        enc_in, dec_in, dec_tgt, _, _ = _build_arrays(fra_txt, num_samples)
        full_ds = Seq2SeqDataset(enc_in, dec_in, dec_tgt)
    else:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset not found at '{fra_txt}'. "
                "Point config['data_path'] to the directory containing fra.txt, "
                "or set config['allow_synthetic_data']=True to use random tensors "
                "for smoke-testing only."
            )
        # Synthetic fallback – shapes mirror the fra-eng defaults
        kwargs             = config.get("model_kwargs", {})
        num_encoder_tokens = int(kwargs.get("num_encoder_tokens", 71))
        num_decoder_tokens = int(kwargs.get("num_decoder_tokens", 93))
        n, T_e, T_d        = 200, 20, 25
        enc_in  = torch.randn(n, T_e, num_encoder_tokens)
        dec_in  = torch.randn(n, T_d, num_decoder_tokens)
        dec_tgt = torch.randn(n, T_d, num_decoder_tokens)
        full_ds = torch.utils.data.TensorDataset(enc_in, dec_in, dec_tgt)

    n_total = len(full_ds)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

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
    optimizer,          # noqa: ARG001  (held by FL runtime; not called here)
    config: dict,
) -> torch.Tensor:
    """One forward pass only. Returns the loss tensor WITH grad attached.

    backward() and optimizer.step() are intentionally NOT called here;
    the FL runtime is responsible for those.
    """
    device = next(model.parameters()).device

    enc_in, dec_in, dec_tgt = batch
    enc_in  = enc_in.to(device)
    dec_in  = dec_in.to(device)
    dec_tgt = dec_tgt.to(device)

    # Forward pass → softmax probabilities  (B, T_dec, num_decoder_tokens)
    predictions = model(enc_in, dec_in)

    # Categorical cross-entropy matching the original Keras loss:
    #   loss = -mean_over_batch( sum_over_classes( target * log(pred) ) )
    # A small epsilon guards against log(0).
    eps  = 1e-7
    loss = -(dec_tgt * torch.log(predictions + eps)).sum(dim=-1).mean()
    return loss
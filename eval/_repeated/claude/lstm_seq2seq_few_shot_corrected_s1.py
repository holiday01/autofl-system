"""
Auto-generated FL client module.
Original script: character-level LSTM seq2seq (English → French translation).

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


class Seq2SeqCharDataset(Dataset):
    """
    Character-level seq2seq dataset from the Anki fra-eng corpus.
    Falls back to a synthetic vocabulary when data_path is not found.
    """

    _SYN_ENC_TOKENS = 30
    _SYN_DEC_TOKENS = 35
    _SYN_ENC_LEN = 20
    _SYN_DEC_LEN = 25
    _SYN_N = 200

    def __init__(self, data_path: str = "fra.txt", num_samples: int = 10000):
        if os.path.isfile(data_path):
            self._load_real(data_path, num_samples)
        else:
            self._load_synthetic()

    def _load_real(self, data_path: str, num_samples: int):
        input_texts, target_texts = [], []
        input_characters, target_characters = set(), set()

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
            input_characters.update(input_text)
            target_characters.update(target_text)

        input_characters = sorted(input_characters)
        target_characters = sorted(target_characters)

        self.input_token_index = {c: i for i, c in enumerate(input_characters)}
        self.target_token_index = {c: i for i, c in enumerate(target_characters)}
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max(len(t) for t in input_texts)
        self.max_decoder_seq_length = max(len(t) for t in target_texts)

        n = len(input_texts)
        enc_in = np.zeros((n, self.max_encoder_seq_length, self.num_encoder_tokens), dtype=np.float32)
        dec_in = np.zeros((n, self.max_decoder_seq_length, self.num_decoder_tokens), dtype=np.float32)
        dec_tgt = np.zeros((n, self.max_decoder_seq_length, self.num_decoder_tokens), dtype=np.float32)

        space_enc = self.input_token_index.get(" ", 0)
        space_dec = self.target_token_index.get(" ", 0)

        for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(inp):
                enc_in[i, t, self.input_token_index[char]] = 1.0
            enc_in[i, len(inp) :, space_enc] = 1.0
            for t, char in enumerate(tgt):
                dec_in[i, t, self.target_token_index[char]] = 1.0
                if t > 0:
                    dec_tgt[i, t - 1, self.target_token_index[char]] = 1.0
            dec_in[i, len(tgt) :, space_dec] = 1.0
            dec_tgt[i, len(tgt) - 1 :, space_dec] = 1.0

        self.encoder_input_data = torch.from_numpy(enc_in)
        self.decoder_input_data = torch.from_numpy(dec_in)
        self.decoder_target_data = torch.from_numpy(dec_tgt)

    def _load_synthetic(self):
        self.num_encoder_tokens = self._SYN_ENC_TOKENS
        self.num_decoder_tokens = self._SYN_DEC_TOKENS
        self.max_encoder_seq_length = self._SYN_ENC_LEN
        self.max_decoder_seq_length = self._SYN_DEC_LEN
        n = self._SYN_N
        rng = np.random.default_rng(42)

        def _random_onehot(n, seq_len, n_tokens):
            idx = rng.integers(0, n_tokens, size=(n, seq_len))
            arr = np.zeros((n, seq_len, n_tokens), dtype=np.float32)
            for i in range(n):
                for t in range(seq_len):
                    arr[i, t, idx[i, t]] = 1.0
            return arr

        self.encoder_input_data = torch.from_numpy(
            _random_onehot(n, self._SYN_ENC_LEN, self._SYN_ENC_TOKENS)
        )
        self.decoder_input_data = torch.from_numpy(
            _random_onehot(n, self._SYN_DEC_LEN, self._SYN_DEC_TOKENS)
        )
        self.decoder_target_data = torch.from_numpy(
            _random_onehot(n, self._SYN_DEC_LEN, self._SYN_DEC_TOKENS)
        )

    def __len__(self):
        return self.encoder_input_data.shape[0]

    def __getitem__(self, idx):
        return (
            self.encoder_input_data[idx],
            self.decoder_input_data[idx],
            self.decoder_target_data[idx],
        )


class Seq2SeqModel(nn.Module):
    """
    Encoder-decoder LSTM seq2seq model (teacher-forcing during training).
    num_encoder_tokens / num_decoder_tokens must match the dataset vocabulary sizes.
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
        self, encoder_input: torch.Tensor, decoder_input: torch.Tensor
    ) -> torch.Tensor:
        _, (h, c) = self.encoder_lstm(encoder_input)
        dec_out, _ = self.decoder_lstm(decoder_input, (h, c))
        return self.decoder_dense(dec_out)  # (batch, seq_len, num_decoder_tokens)


# ── FL Interface ────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    return Seq2SeqModel(**kwargs)


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

    logits = model(encoder_input, decoder_input)
    # Targets are one-hot (soft), so use manual categorical cross-entropy
    # rather than nn.CrossEntropyLoss (which expects integer class indices).
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(decoder_target * log_probs).sum(dim=-1).mean()
    return loss
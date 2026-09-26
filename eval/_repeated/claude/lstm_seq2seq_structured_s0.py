import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split


# ────────────────────────────────────────────────────────────────────────────
# Model  (Keras → PyTorch conversion)
# Encoder LSTM → final (h, c) states → Decoder LSTM → Dense softmax
# ────────────────────────────────────────────────────────────────────────────

class Seq2SeqModel(nn.Module):
    """
    Character-level encoder-decoder LSTM sequence-to-sequence model.
    Direct PyTorch equivalent of the original Keras architecture:
      - Encoder LSTM (returns final hidden/cell states only)
      - Decoder LSTM initialised with encoder states (teacher-forced training)
      - Linear projection to num_decoder_tokens with log-softmax loss
    """

    def __init__(
        self,
        num_encoder_tokens: int,
        num_decoder_tokens: int,
        latent_dim: int = 256,
    ) -> None:
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
        self.decoder_dense = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_input : (B, T_enc, num_encoder_tokens)  — one-hot
            decoder_input : (B, T_dec, num_decoder_tokens)  — one-hot teacher-forced
        Returns:
            logits : (B, T_dec, num_decoder_tokens)
        """
        # Encode: keep only final (h, c); discard output sequence
        _, (h, c) = self.encoder_lstm(encoder_input)  # h, c: (1, B, latent_dim)

        # Decode with encoder states as initial state
        decoder_out, _ = self.decoder_lstm(decoder_input, (h, c))  # (B, T_dec, latent_dim)

        logits = self.decoder_dense(decoder_out)  # (B, T_dec, num_decoder_tokens)
        return logits


# ────────────────────────────────────────────────────────────────────────────
# Real dataset
# ────────────────────────────────────────────────────────────────────────────

class Seq2SeqDataset(Dataset):
    """
    Loads the fra-eng parallel corpus (fra.txt) and returns one-hot encoded
    (encoder_input, decoder_input, decoder_target) triples — matching the
    original script's vectorisation logic exactly.
    """

    def __init__(self, data_path: str, num_samples: int = 10000) -> None:
        input_texts: list[str] = []
        target_texts: list[str] = []
        input_characters: set[str] = set()
        target_characters: set[str] = set()

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
            for char in input_text:
                input_characters.add(char)
            for char in target_text:
                target_characters.add(char)

        input_characters_list = sorted(input_characters)
        target_characters_list = sorted(target_characters)

        self.num_encoder_tokens = len(input_characters_list)
        self.num_decoder_tokens = len(target_characters_list)
        self.max_encoder_seq_length = max(len(t) for t in input_texts)
        self.max_decoder_seq_length = max(len(t) for t in target_texts)

        input_token_index = {char: i for i, char in enumerate(input_characters_list)}
        target_token_index = {char: i for i, char in enumerate(target_characters_list)}
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index

        N = len(input_texts)
        enc = np.zeros(
            (N, self.max_encoder_seq_length, self.num_encoder_tokens), dtype=np.float32
        )
        dec_in = np.zeros(
            (N, self.max_decoder_seq_length, self.num_decoder_tokens), dtype=np.float32
        )
        dec_tgt = np.zeros(
            (N, self.max_decoder_seq_length, self.num_decoder_tokens), dtype=np.float32
        )

        space_enc_idx = input_token_index.get(" ", 0)
        space_dec_idx = target_token_index.get(" ", 0)

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            t_enc = 0
            for t_enc, char in enumerate(input_text):
                enc[i, t_enc, input_token_index[char]] = 1.0
            enc[i, t_enc + 1 :, space_enc_idx] = 1.0

            t_dec = 0
            for t_dec, char in enumerate(target_text):
                dec_in[i, t_dec, target_token_index[char]] = 1.0
                if t_dec > 0:
                    dec_tgt[i, t_dec - 1, target_token_index[char]] = 1.0
            dec_in[i, t_dec + 1 :, space_dec_idx] = 1.0
            dec_tgt[i, t_dec :, space_dec_idx] = 1.0

        self.encoder_input_data = torch.from_numpy(enc)
        self.decoder_input_data = torch.from_numpy(dec_in)
        self.decoder_target_data = torch.from_numpy(dec_tgt)

    def __len__(self) -> int:
        return self.encoder_input_data.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.encoder_input_data[idx],
            self.decoder_input_data[idx],
            self.decoder_target_data[idx],
        )


# ────────────────────────────────────────────────────────────────────────────
# Synthetic fallback dataset (gated — see build_dataloader)
# ────────────────────────────────────────────────────────────────────────────

class _SyntheticSeq2SeqDataset(Dataset):
    """
    Random one-hot stand-in used ONLY when allow_synthetic_data=True.
    Shapes are kept consistent with the real dataset vocabulary dimensions.
    """

    def __init__(
        self,
        num_encoder_tokens: int,
        num_decoder_tokens: int,
        max_encoder_seq_length: int,
        max_decoder_seq_length: int,
        num_samples: int = 200,
    ) -> None:
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def _one_hot(T: int, C: int) -> torch.Tensor:
        indices = torch.randint(0, C, (T,))
        x = torch.zeros(T, C)
        x.scatter_(1, indices.unsqueeze(1), 1.0)
        return x

    def __getitem__(self, idx: int):
        enc = self._one_hot(self.max_encoder_seq_length, self.num_encoder_tokens)
        dec_in = self._one_hot(self.max_decoder_seq_length, self.num_decoder_tokens)
        dec_tgt = self._one_hot(self.max_decoder_seq_length, self.num_decoder_tokens)
        return enc, dec_in, dec_tgt


# ────────────────────────────────────────────────────────────────────────────
# FL entry points
# ────────────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the Seq2SeqModel.

    Relevant config keys (all inside config["model_kwargs"]):
        num_encoder_tokens  (default 71  — typical for fra-eng English vocab)
        num_decoder_tokens  (default 93  — typical for fra-eng French vocab)
        latent_dim          (default 256)
    """
    kwargs = config.get("model_kwargs", {})
    return Seq2SeqModel(
        num_encoder_tokens=kwargs.get("num_encoder_tokens", 71),
        num_decoder_tokens=kwargs.get("num_decoder_tokens", 93),
        latent_dim=kwargs.get("latent_dim", 256),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ("train" or "val").

    Relevant config keys:
        data_path                  path to fra.txt or its parent directory
        local.batch_size           (default 16)
        num_samples                rows to read from fra.txt (default 10000)
        val_fraction               fraction held out for validation (default 0.2)
        seed                       random_split seed (default 42)
        allow_synthetic_data       must be True to use synthetic data (default False)
        model_kwargs.*             forwarded to _SyntheticSeq2SeqDataset when real
                                   data is unavailable and synthetic is allowed
    """
    batch_size: int = config.get("local", {}).get("batch_size", 16)
    data_path: str = config.get("data_path", ".")
    num_samples: int = config.get("num_samples", 10000)
    val_fraction: float = config.get("val_fraction", 0.2)
    seed: int = config.get("seed", 42)

    # ── Resolve path to fra.txt ──────────────────────────────────────────────
    fra_txt: str | None = None
    if os.path.isfile(data_path):
        fra_txt = data_path
    elif os.path.isdir(data_path):
        for candidate in ("fra.txt", os.path.join("fra-eng", "fra.txt")):
            p = os.path.join(data_path, candidate)
            if os.path.exists(p):
                fra_txt = p
                break

    # ── Build dataset ────────────────────────────────────────────────────────
    if fra_txt is not None:
        dataset: Dataset = Seq2SeqDataset(fra_txt, num_samples=num_samples)
    else:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset (fra.txt) not found at data_path='{data_path}'. "
                "Download it from http://www.manythings.org/anki/fra-eng.zip, "
                "or set config['allow_synthetic_data'] = True to use random "
                "one-hot data for smoke-testing only."
            )
        mkw = config.get("model_kwargs", {})
        dataset = _SyntheticSeq2SeqDataset(
            num_encoder_tokens=mkw.get("num_encoder_tokens", 71),
            num_decoder_tokens=mkw.get("num_decoder_tokens", 93),
            max_encoder_seq_length=mkw.get("max_encoder_seq_length", 16),
            max_decoder_seq_length=mkw.get("max_decoder_seq_length", 59),
            num_samples=num_samples,
        )

    # ── Train / val split ────────────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    chosen_ds = train_ds if split == "train" else val_ds
    return DataLoader(
        chosen_ds,
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
    """
    Single forward pass — returns the scalar loss WITH grad attached.
    The FL runtime is responsible for loss.backward() and optimizer.step().

    Loss: categorical cross-entropy (matches Keras 'categorical_crossentropy')
          -sum(y_true * log_softmax(logits), dim=-1).mean()
    """
    device = next(model.parameters()).device

    encoder_input, decoder_input, decoder_target = batch
    encoder_input = encoder_input.to(device)    # (B, T_enc, num_encoder_tokens)
    decoder_input = decoder_input.to(device)    # (B, T_dec, num_decoder_tokens)
    decoder_target = decoder_target.to(device)  # (B, T_dec, num_decoder_tokens) one-hot

    logits = model(encoder_input, decoder_input)  # (B, T_dec, num_decoder_tokens)

    # Categorical cross-entropy with one-hot targets:
    #   loss = -mean_over(B,T) [ sum_over_C( y_true * log_softmax(logits) ) ]
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(decoder_target * log_probs).sum(dim=-1).mean()

    return loss  # grad attached; caller invokes backward() + optimizer.step()
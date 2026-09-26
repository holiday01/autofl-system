import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ── Preserved model architecture (Keras LSTM encoder-decoder → PyTorch) ───────

class Seq2SeqModel(nn.Module):
    """
    Character-level encoder-decoder LSTM.

    Mirrors the original Keras architecture:
      Encoder LSTM : (B, enc_T, num_encoder_tokens) → final states (h, c)
      Decoder LSTM : conditioned on encoder states,
                     (B, dec_T, num_decoder_tokens) → hidden sequence
      Dense        : hidden → logits over num_decoder_tokens

    Returns raw logits (no softmax); the training loss applies log-softmax
    internally via F.cross_entropy for numerical stability.
    """

    def __init__(
        self,
        num_encoder_tokens: int,
        num_decoder_tokens: int,
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
        self.decoder_dense = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_input  (B, enc_T, num_encoder_tokens)
            decoder_input  (B, dec_T, num_decoder_tokens)
        Returns:
            logits         (B, dec_T, num_decoder_tokens)
        """
        _, (h, c) = self.encoder_lstm(encoder_input)
        decoder_out, _ = self.decoder_lstm(decoder_input, (h, c))
        return self.decoder_dense(decoder_out)


# ── Dataset ────────────────────────────────────────────────────────────────────

class Seq2SeqDataset(Dataset):
    """
    Reads *data_path/fra.txt* (from http://www.manythings.org/anki/fra-eng.zip)
    and builds three one-hot float32 tensors per sample:
        encoder_input  (max_enc_len, num_encoder_tokens)
        decoder_input  (max_dec_len, num_decoder_tokens)
        decoder_target (max_dec_len, num_decoder_tokens)

    Preprocessing replicates the original script exactly (tab / newline
    start-stop tokens, padding with the space character).
    """

    def __init__(self, data_path: str, num_samples: int = 10000):
        fra_txt = os.path.join(data_path, "fra.txt")
        if not os.path.isfile(fra_txt):
            raise FileNotFoundError(
                f"Expected dataset file at: {fra_txt}. "
                "Download fra-eng.zip from http://www.manythings.org/anki/ and "
                "unzip it into data_path, or set config['allow_synthetic_data'] = True."
            )

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
            input_characters.update(input_text)
            target_characters.update(target_text)

        input_characters_list = sorted(input_characters)
        target_characters_list = sorted(target_characters)

        self.num_encoder_tokens = len(input_characters_list)
        self.num_decoder_tokens = len(target_characters_list)
        max_enc_len = max(len(t) for t in input_texts)
        max_dec_len = max(len(t) for t in target_texts)

        input_token_index = {ch: i for i, ch in enumerate(input_characters_list)}
        target_token_index = {ch: i for i, ch in enumerate(target_characters_list)}

        n = len(input_texts)
        enc_in = np.zeros((n, max_enc_len, self.num_encoder_tokens), dtype="float32")
        dec_in = np.zeros((n, max_dec_len, self.num_decoder_tokens), dtype="float32")
        dec_tgt = np.zeros((n, max_dec_len, self.num_decoder_tokens), dtype="float32")

        space_enc_idx = input_token_index.get(" ", 0)
        space_dec_idx = target_token_index.get(" ", 0)

        for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
            last_enc_t = 0
            for t, ch in enumerate(inp):
                enc_in[i, t, input_token_index[ch]] = 1.0
                last_enc_t = t
            enc_in[i, last_enc_t + 1:, space_enc_idx] = 1.0

            last_dec_t = 0
            for t, ch in enumerate(tgt):
                dec_in[i, t, target_token_index[ch]] = 1.0
                if t > 0:
                    dec_tgt[i, t - 1, target_token_index[ch]] = 1.0
                last_dec_t = t
            dec_in[i, last_dec_t + 1:, space_dec_idx] = 1.0
            dec_tgt[i, last_dec_t:, space_dec_idx] = 1.0

        self.encoder_input = torch.from_numpy(enc_in)
        self.decoder_input = torch.from_numpy(dec_in)
        self.decoder_target = torch.from_numpy(dec_tgt)

    def __len__(self) -> int:
        return self.encoder_input.size(0)

    def __getitem__(self, idx):
        return (
            self.encoder_input[idx],
            self.decoder_input[idx],
            self.decoder_target[idx],
        )


# ── FL API ─────────────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """
    Instantiate and return the Seq2SeqModel.

    Recognised config['model_kwargs'] keys
    (all optional; defaults match the typical fra-eng vocabulary):
        num_encoder_tokens  int   default 71
        num_decoder_tokens  int   default 93
        latent_dim          int   default 256
    """
    kwargs = config.get("model_kwargs", {})
    return Seq2SeqModel(
        num_encoder_tokens=kwargs.get("num_encoder_tokens", 71),
        num_decoder_tokens=kwargs.get("num_decoder_tokens", 93),
        latent_dim=kwargs.get("latent_dim", 256),
    )


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Return a DataLoader for the requested split ('train' or 'val').

    Config keys consumed:
        data_path                        str    directory containing fra.txt  (default ".")
        local.batch_size                 int    default 16
        num_samples                      int    max lines to read             (default 10000)
        val_fraction                     float  fraction held out for val     (default 0.2)
        allow_synthetic_data             bool   must be True to use fallback  (default False)
        model_kwargs.num_encoder_tokens  int    used only for synthetic data  (default 71)
        model_kwargs.num_decoder_tokens  int    used only for synthetic data  (default 93)

    If fra.txt is missing and allow_synthetic_data is False a FileNotFoundError
    is raised immediately — synthetic tensors are never used silently.
    """
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    num_samples = config.get("num_samples", 10000)
    val_fraction = config.get("val_fraction", 0.2)

    try:
        dataset: Dataset = Seq2SeqDataset(data_path=data_path, num_samples=num_samples)
    except FileNotFoundError as exc:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"{exc}\n"
                "Real data is unavailable and config['allow_synthetic_data'] is False. "
                "Set it to True only for smoke-testing purposes."
            ) from exc

        # ── Synthetic-data fallback (smoke-test only) ───────────────────────
        kwargs = config.get("model_kwargs", {})
        _num_enc = kwargs.get("num_encoder_tokens", 71)
        _num_dec = kwargs.get("num_decoder_tokens", 93)
        _enc_seq_len = 16
        _dec_seq_len = 20
        _n = num_samples

        class _SyntheticSeq2SeqDataset(Dataset):
            def __init__(self):
                rng = torch.Generator().manual_seed(0)
                self._enc = torch.randn(
                    _n, _enc_seq_len, _num_enc, generator=rng
                )
                self._dec_in = torch.randn(
                    _n, _dec_seq_len, _num_dec, generator=rng
                )
                raw_tgt = torch.randint(
                    0, _num_dec, (_n, _dec_seq_len), generator=rng
                )
                self._dec_tgt = F.one_hot(raw_tgt, num_classes=_num_dec).float()

            def __len__(self):
                return _n

            def __getitem__(self, idx):
                return self._enc[idx], self._dec_in[idx], self._dec_tgt[idx]

        dataset = _SyntheticSeq2SeqDataset()

    n_total = len(dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    train_subset, val_subset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    chosen = train_subset if split == "train" else val_subset
    return DataLoader(
        chosen,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=False,
    )


def train_step(
    model: nn.Module,
    batch: tuple,
    optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Single forward pass.  Returns the loss tensor WITH grad attached.
    Does NOT call loss.backward() or optimizer.step() — the FL runtime
    is responsible for both.

    Batch layout (matches Seq2SeqDataset.__getitem__):
        encoder_input  (B, enc_T, num_encoder_tokens)  float32
        decoder_input  (B, dec_T, num_decoder_tokens)  float32
        decoder_target (B, dec_T, num_decoder_tokens)  float32 one-hot

    Loss: categorical cross-entropy (F.cross_entropy on logits),
    matching the original model.compile(loss='categorical_crossentropy').
    """
    device = next(model.parameters()).device

    encoder_input, decoder_input, decoder_target = batch
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    decoder_target = decoder_target.to(device)

    # Forward pass → logits (B, dec_T, num_decoder_tokens)
    logits = model(encoder_input, decoder_input)

    # One-hot targets → class indices for F.cross_entropy
    target_indices = decoder_target.argmax(dim=-1)   # (B, dec_T)

    B, T, C = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, C),
        target_indices.reshape(B * T).long(),
    )
    return loss
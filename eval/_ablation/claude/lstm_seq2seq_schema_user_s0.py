import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


class Seq2SeqModel(nn.Module):
    def __init__(self, num_encoder_tokens, num_decoder_tokens, latent_dim=256):
        super().__init__()
        self.encoder_lstm = nn.LSTM(num_encoder_tokens, latent_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(num_decoder_tokens, latent_dim, batch_first=True)
        self.decoder_dense = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(self, encoder_input, decoder_input):
        _, (h, c) = self.encoder_lstm(encoder_input)
        decoder_output, _ = self.decoder_lstm(decoder_input, (h, c))
        return self.decoder_dense(decoder_output)


class _Seq2SeqDataset(Dataset):
    def __init__(self, encoder_data, decoder_input_data, decoder_target_data):
        self.enc = torch.tensor(encoder_data, dtype=torch.float32)
        self.dec_in = torch.tensor(decoder_input_data, dtype=torch.float32)
        self.dec_tgt = torch.tensor(decoder_target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        return self.enc[idx], self.dec_in[idx], self.dec_tgt[idx]


def _load_fra_eng_data(data_path, num_samples=10000):
    fra_txt = os.path.join(data_path, "fra.txt")
    if not os.path.exists(fra_txt):
        raise FileNotFoundError(f"Expected data file not found: {fra_txt}")

    input_texts, target_texts = [], []
    input_characters, target_characters = set(), set()

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
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max(len(t) for t in input_texts)
    max_decoder_seq_length = max(len(t) for t in target_texts)

    input_token_index = {char: i for i, char in enumerate(input_characters)}
    target_token_index = {char: i for i, char in enumerate(target_characters)}

    n = len(input_texts)
    encoder_input_data = np.zeros((n, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    decoder_input_data = np.zeros((n, max_decoder_seq_length, num_decoder_tokens), dtype="float32")
    decoder_target_data = np.zeros((n, max_decoder_seq_length, num_decoder_tokens), dtype="float32")

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[i, t :, target_token_index[" "]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data, {
        "num_encoder_tokens": num_encoder_tokens,
        "num_decoder_tokens": num_decoder_tokens,
        "max_encoder_seq_length": max_encoder_seq_length,
        "max_decoder_seq_length": max_decoder_seq_length,
    }


def build_model(config: dict) -> nn.Module:
    kwargs = config.get("model_kwargs", {})
    num_encoder_tokens = kwargs.get("num_encoder_tokens", 71)
    num_decoder_tokens = kwargs.get("num_decoder_tokens", 93)
    latent_dim = kwargs.get("latent_dim", 256)
    return Seq2SeqModel(num_encoder_tokens, num_decoder_tokens, latent_dim)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    kwargs = config.get("model_kwargs", {})
    num_samples = kwargs.get("num_samples", 10000)

    try:
        enc_in, dec_in, dec_tgt, _ = _load_fra_eng_data(data_path, num_samples)
        dataset = _Seq2SeqDataset(enc_in, dec_in, dec_tgt)
    except FileNotFoundError as exc:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset not found at '{data_path}/fra.txt'. "
                "Set config['allow_synthetic_data']=True to fall back to synthetic data."
            ) from exc

        num_encoder_tokens = kwargs.get("num_encoder_tokens", 71)
        num_decoder_tokens = kwargs.get("num_decoder_tokens", 93)
        max_enc_len = kwargs.get("max_encoder_seq_length", 16)
        max_dec_len = kwargs.get("max_decoder_seq_length", 59)
        n = int(num_samples)

        rng = np.random.default_rng()
        enc_in = np.zeros((n, max_enc_len, num_encoder_tokens), dtype="float32")
        dec_in = np.zeros((n, max_dec_len, num_decoder_tokens), dtype="float32")
        dec_tgt = np.zeros((n, max_dec_len, num_decoder_tokens), dtype="float32")
        enc_in[
            np.arange(n)[:, None],
            np.arange(max_enc_len)[None, :],
            rng.integers(0, num_encoder_tokens, (n, max_enc_len)),
        ] = 1.0
        dec_in[
            np.arange(n)[:, None],
            np.arange(max_dec_len)[None, :],
            rng.integers(0, num_decoder_tokens, (n, max_dec_len)),
        ] = 1.0
        dec_tgt[
            np.arange(n)[:, None],
            np.arange(max_dec_len)[None, :],
            rng.integers(0, num_decoder_tokens, (n, max_dec_len)),
        ] = 1.0
        dataset = _Seq2SeqDataset(enc_in, dec_in, dec_tgt)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    chosen = train_ds if split == "train" else val_ds
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model: nn.Module, batch, optimizer, config: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    encoder_input, decoder_input, decoder_target = batch
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    decoder_target = decoder_target.to(device)

    logits = model(encoder_input, decoder_input)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(decoder_target * log_probs).sum(dim=-1).mean()
    return loss
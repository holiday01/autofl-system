import numpy as np
import keras
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


def _load_and_vectorize(data_path, num_samples=10000):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
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

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max(len(txt) for txt in input_texts)
    max_decoder_seq_length = max(len(txt) for txt in target_texts)

    input_token_index = {char: i for i, char in enumerate(input_characters)}
    target_token_index = {char: i for i, char in enumerate(target_characters)}

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

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

    return encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens


class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_input, decoder_input, decoder_target):
        self.encoder_input = torch.tensor(encoder_input, dtype=torch.float32)
        self.decoder_input = torch.tensor(decoder_input, dtype=torch.float32)
        self.decoder_target = torch.tensor(decoder_target, dtype=torch.float32)

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[idx], self.decoder_target[idx]


class Seq2SeqModel(nn.Module):
    def __init__(self, num_encoder_tokens, num_decoder_tokens, latent_dim=256):
        super().__init__()
        self.encoder_lstm = nn.LSTM(num_encoder_tokens, latent_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(num_decoder_tokens, latent_dim, batch_first=True)
        self.decoder_dense = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(self, encoder_input, decoder_input):
        _, (state_h, state_c) = self.encoder_lstm(encoder_input)
        decoder_output, _ = self.decoder_lstm(decoder_input, (state_h, state_c))
        return F.softmax(self.decoder_dense(decoder_output), dim=-1)


def build_model(config):
    latent_dim = config.get("latent_dim", 256)
    num_encoder_tokens = config.get("num_encoder_tokens")
    num_decoder_tokens = config.get("num_decoder_tokens")
    if num_encoder_tokens is None or num_decoder_tokens is None:
        data_path = os.path.join(config.get("data_path", "."), "fra.txt")
        num_samples = config.get("num_samples", 10000)
        _, _, _, num_encoder_tokens, num_decoder_tokens = _load_and_vectorize(data_path, num_samples)
    return Seq2SeqModel(num_encoder_tokens, num_decoder_tokens, latent_dim)


def build_dataloader(config, split="train"):
    data_path = os.path.join(config.get("data_path", "."), "fra.txt")
    batch_size = config.get("local", {}).get("batch_size", 16)
    num_samples = config.get("num_samples", 10000)
    val_fraction = config.get("val_fraction", 0.2)

    enc_in, dec_in, dec_tgt, _, _ = _load_and_vectorize(data_path, num_samples)
    dataset = Seq2SeqDataset(enc_in, dec_in, dec_tgt)

    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    chosen = train_dataset if split == "train" else val_dataset
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config):
    encoder_input, decoder_input, decoder_target = batch
    optimizer.zero_grad()
    output = model(encoder_input, decoder_input)
    loss = -(decoder_target * torch.log(output + 1e-8)).sum(dim=-1).mean()
    loss.backward()
    optimizer.step()
    return loss.item()
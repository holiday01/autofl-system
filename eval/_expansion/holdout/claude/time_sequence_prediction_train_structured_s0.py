from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, random_split


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class _SequenceDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def build_model(config: dict) -> nn.Module:
    model_kwargs = config.get("model_kwargs", {})
    model = Sequence(**model_kwargs)
    model.double()
    return model


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    batch_size = config.get("local", {}).get("batch_size", 16)
    data_path = config.get("data_path", ".")
    pt_file = os.path.join(data_path, "traindata.pt")

    if os.path.exists(pt_file):
        try:
            data = torch.load(pt_file, weights_only=False)
        except TypeError:
            data = torch.load(pt_file)
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        inputs = data[:, :-1].double()
        targets = data[:, 1:].double()
    else:
        if not config.get("allow_synthetic_data", False):
            raise FileNotFoundError(
                f"Real dataset file not found at '{pt_file}'. "
                "Set config['allow_synthetic_data'] = True to allow synthetic data as a fallback."
            )
        n_sequences = 100
        seq_len = 999
        t = torch.linspace(0, 4 * np.pi, seq_len + 1, dtype=torch.double)
        noise = torch.randn(n_sequences, seq_len + 1, dtype=torch.double) * 0.05
        base = torch.sin(t).unsqueeze(0).expand(n_sequences, -1).clone() + noise
        inputs = base[:, :-1]
        targets = base[:, 1:]

    dataset = _SequenceDataset(inputs, targets)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    chosen = train_set if split == "train" else val_set
    return DataLoader(chosen, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config: dict) -> torch.Tensor:
    # NOTE: Sequence.forward() allocates hidden states on CPU (original architecture
    # preserved verbatim). This client therefore targets CPU, which is the standard
    # deployment target for float64 LSTM models.
    device = next(model.parameters()).device
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    criterion = nn.MSELoss()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    return loss
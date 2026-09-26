from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


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
            outputs.append(output)
        for _ in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        return torch.cat(outputs, dim=1)


def build_model(config):
    model = Sequence()
    model.double()
    return model


def build_dataloader(config, split):
    data_path = config.get("data_path", "traindata.pt")
    batch_size = config.get("batch_size", 32)

    data = torch.load(data_path)

    if split == "train":
        input_data = torch.from_numpy(data[3:, :-1])
        target_data = torch.from_numpy(data[3:, 1:])
    elif split in ("val", "test"):
        input_data = torch.from_numpy(data[:3, :-1])
        target_data = torch.from_numpy(data[:3, 1:])
    else:
        raise ValueError(f"Unknown split: {split!r}")

    dataset = TensorDataset(input_data, target_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


def train_step(model, batch, optimizer, config):
    inputs, targets = batch
    criterion = nn.MSELoss()

    def closure():
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    return loss.item()
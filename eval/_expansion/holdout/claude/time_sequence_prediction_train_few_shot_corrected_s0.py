"""
Auto-generated FL client module.
Original script: LSTM time-series prediction (Sequence model).

Exposes:
  build_model(config)                   -> nn.Module
  build_dataloader(config, split)       -> DataLoader
  train_step(model, batch, opt, config) -> loss tensor (with grad_fn)

CONTRACT:
  - train_step performs ONE forward pass and returns the raw loss tensor.
  - The returned tensor MUST have grad_fn attached (do NOT call .detach()).
  - Do NOT call loss.backward() inside train_step.
  - Do NOT call optimizer.step() or optimizer.zero_grad() inside train_step.
  - Do NOT call .item() on the returned loss.
  The FL runtime owns backward(), step(), and metric extraction.

NOTE on LBFGS: the original script used LBFGS with a closure over the full
dataset.  The FL runtime drives optimizer.step() externally, so first-order
optimizers (Adam, SGD) are the expected default here.  If the runtime supports
LBFGS-style closures, configure it accordingly outside this module.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class Sequence(nn.Module):
    def __init__(self, hidden_size: int = 51):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(1, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input, future: int = 0):
        outputs = []
        h_t  = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        c_t  = torch.zeros_like(h_t)
        h_t2 = torch.zeros_like(h_t)
        c_t2 = torch.zeros_like(h_t)

        for input_t in input.split(1, dim=1):
            h_t,  c_t  = self.lstm1(input_t, (h_t,  c_t))
            h_t2, c_t2 = self.lstm2(h_t,    (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        for _ in range(future):
            h_t,  c_t  = self.lstm1(output, (h_t,  c_t))
            h_t2, c_t2 = self.lstm2(h_t,   (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        return torch.cat(outputs, dim=1)


class TimeSeriesDataset(Dataset):
    """Each sample is one (input_seq, target_seq) row from the raw data tensor."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        assert inputs.shape[0] == targets.shape[0]
        self.inputs  = inputs.float()
        self.targets = targets.float()

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    @classmethod
    def from_file(cls, path: str, split: str = "train", test_rows: int = 3):
        data = torch.load(path, weights_only=False)
        if split == "train":
            inp = torch.from_numpy(data[test_rows:, :-1])
            tgt = torch.from_numpy(data[test_rows:,  1:])
        else:
            inp = torch.from_numpy(data[:test_rows, :-1])
            tgt = torch.from_numpy(data[:test_rows,  1:])
        return cls(inp, tgt)

    @classmethod
    def synthetic(cls, n_sequences: int = 100, seq_len: int = 999):
        """Fallback synthetic dataset for testing when data file is absent."""
        t = torch.linspace(0, 4 * 3.14159, seq_len + 1)
        rows = torch.stack([torch.sin(t + i * 0.5) for i in range(n_sequences)])
        return cls(rows[:, :-1], rows[:, 1:])


# ── FL Interface ─────────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    hidden_size = config.get("model_kwargs", {}).get("hidden_size", 51)
    return Sequence(hidden_size=hidden_size)


def build_dataloader(config: dict, split: str = "train") -> DataLoader:
    local       = config.get("local", {})
    batch_size  = local.get("batch_size",  config.get("batch_size",  16))
    num_workers = local.get("num_workers", config.get("num_workers",  0))
    pin_memory  = local.get("pin_memory",  True)

    data_path = config.get("data_path", "traindata.pt")
    test_rows = config.get("test_rows", 3)

    if os.path.isfile(data_path):
        dataset = TimeSeriesDataset.from_file(data_path, split=split, test_rows=test_rows)
    else:
        # synthetic fallback
        full = TimeSeriesDataset.synthetic(
            n_sequences=config.get("n_sequences", 100),
            seq_len=config.get("seq_len", 999),
        )
        val_ratio = config.get("val_ratio", 0.1)
        n_val   = max(1, int(len(full) * val_ratio))
        n_train = len(full) - n_val
        train_ds, val_ds = random_split(
            full, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.get("seed", 0)),
        )
        dataset = train_ds if split == "train" else val_ds

    return DataLoader(
        dataset,
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

    if isinstance(batch, (list, tuple)):
        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        inputs, targets = batch[0], batch[1]
    elif isinstance(batch, dict):
        batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        inputs  = batch.get("input",  batch.get("x"))
        targets = batch.get("target", batch.get("y"))
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # unsqueeze last dim if bare (batch, seq_len) — LSTMCell expects (batch, 1) per step
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(-1)

    outputs = model(inputs)
    outputs = outputs.squeeze(-1)

    loss = nn.MSELoss()(outputs, targets)
    return loss
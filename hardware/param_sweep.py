"""
Hardware parameter sweep: batch_size x learning_rate grid (CPU mode).
Uses a 500-sample MNIST subset + tiny MLP for fast CPU timing.
Records: batch_size, lr, avg_loss_round2, steps_per_sec, total_elapsed_sec.
Saves to <REPO>/results/hardware_sweep.csv
"""

import sys
import time
import csv
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hardware.detector import detect

# ── grid ─────────────────────────────────────────────────────────────────────
BATCH_SIZES    = [4, 8, 16, 32]
LEARNING_RATES = [1e-2, 1e-3, 3e-4, 1e-4]
FL_ROUNDS      = 2
LOCAL_EPOCHS   = 1
GRAD_ACCUM     = 1
MAX_TRAIN_SAMPLES = 500   # tiny subset for fast CPU sweep

DATA_PATH = str(PROJECT_ROOT / "data")
OUT_CSV   = PROJECT_ROOT / "results/hardware_sweep.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── detector baseline ─────────────────────────────────────────────────────────
profile = detect()
det_bs  = profile.suggested_batch_size
det_lr  = 3e-4   # CPU path default (detector doesn't emit lr directly)
print(f"[detector] suggested batch_size={det_bs}, use_amp={profile.suggested_use_amp}")
print(f"[detector] (using lr={det_lr:.0e} as CPU default)\n")


# ── tiny MLP (fast on CPU, same input/output dims as MNIST) ──────────────────
class TinyMLP(nn.Module):
    """Two-layer MLP: 784 -> 128 -> 10. ~100k params, very fast on CPU."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


# ── load dataset once ─────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

print(f"Loading MNIST subset ({MAX_TRAIN_SAMPLES} samples) ...", flush=True)
try:
    full_ds  = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    train_ds = Subset(full_ds, list(range(MAX_TRAIN_SAMPLES)))
    print(f"  MNIST loaded OK ({len(full_ds)} total, using {len(train_ds)})\n")
except Exception as e:
    print(f"  MNIST download failed ({e}), using synthetic data")
    data     = torch.randn(MAX_TRAIN_SAMPLES, 1, 28, 28)
    targets  = torch.randint(0, 10, (MAX_TRAIN_SAMPLES,))
    train_ds = TensorDataset(data, targets)
    print(f"  Synthetic dataset: {len(train_ds)} samples\n")

# Shared initial weights (same random seed for fair comparison)
torch.manual_seed(42)
_init_weights = copy.deepcopy(TinyMLP().state_dict())


def run_config(batch_size: int, lr: float) -> dict:
    device = torch.device("cpu")
    model  = TinyMLP().to(device)
    model.load_state_dict(copy.deepcopy(_init_weights))
    global_w = copy.deepcopy(model.state_dict())

    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    round_losses = []
    total_steps  = 0
    t_start = time.time()

    for r in range(FL_ROUNDS):
        model.load_state_dict(copy.deepcopy(global_w))
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        epoch_loss, n_steps = 0.0, 0
        for epoch in range(LOCAL_EPOCHS):
            for step, batch in enumerate(dl):
                if step % GRAD_ACCUM == 0:
                    opt.zero_grad()
                data, target = batch[0].to(device), batch[1].to(device)
                loss = F.nll_loss(model(data), target)
                epoch_loss += loss.item()
                n_steps    += 1
                (loss / GRAD_ACCUM).backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

        round_losses.append(epoch_loss / max(n_steps, 1))
        total_steps += n_steps
        global_w = copy.deepcopy(model.state_dict())

    t_total       = time.time() - t_start
    steps_per_sec = total_steps / max(t_total, 1e-6)

    return {
        "batch_size":        batch_size,
        "lr":                lr,
        "avg_loss_round2":   round(round_losses[-1], 6),
        "steps_per_sec":     round(steps_per_sec, 4),
        "total_elapsed_sec": round(t_total, 2),
    }


# ── sweep ─────────────────────────────────────────────────────────────────────
results       = []
total_configs = len(BATCH_SIZES) * len(LEARNING_RATES)
idx           = 0

for bs in BATCH_SIZES:
    for lr in LEARNING_RATES:
        idx += 1
        print(f"[{idx:2d}/{total_configs}] bs={bs:2d}  lr={lr:.0e} ...", end="  ", flush=True)
        row = run_config(bs, lr)
        results.append(row)
        print(f"loss={row['avg_loss_round2']:.4f}  sps={row['steps_per_sec']:.2f}  t={row['total_elapsed_sec']:.1f}s")

# ── write CSV ─────────────────────────────────────────────────────────────────
fieldnames = ["batch_size", "lr", "avg_loss_round2", "steps_per_sec", "total_elapsed_sec"]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved {len(results)} rows → {OUT_CSV}")
print(f"Detector recommendation: batch_size={det_bs}, lr={det_lr:.0e}")

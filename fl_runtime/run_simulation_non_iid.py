"""
AutoFL non-IID FL simulation.

Compares FedAvg convergence under three data partitionings:
  - IID                       (re-run for direct comparison)
  - Dirichlet alpha = 0.5     (mild skew)
  - Dirichlet alpha = 0.1     (severe skew)
plus a centralized baseline (single trainer on the full dataset, same total
optimizer steps).

Outputs (defaults; the v1 archives in fl_results/ are never overwritten):
  <out-dir>/simulation_non_iid_results.json   (default fl_results/v2/)
  <fig-dir>/fig7_non_iid.{pdf,png}             (default results/figures_v2/)
  <out-dir>/simulation_non_iid_notes.md        (generated from data; never hand-edited)

v2 changes: sample-weighted FedAvg (n_k/n) for both the weights and the
aggregated loss; seed applied to model init; notes report the actual torch
device / AMP state and compute monotonicity from the recorded losses.

The script reuses FLClient/FLServer. Each client is given a `client_indices`
config field that is consumed by a small wrapper around the underlying
`build_dataloader` (we monkey-patch the loaded module so existing
FLClient.local_train logic works unchanged).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fl_runtime"))

from hardware.detector import detect, print_profile
from fl_runtime.client import FLClient
from fl_runtime.server import FLServer
from fl_runtime.non_iid_partition import (
    dirichlet_partition,
    iid_partition,
    partition_summary,
)

# ---------------------------------------------------------------------------
# Hardware / paths
# ---------------------------------------------------------------------------
_HW = detect()
print_profile(_HW)
if not _HW.has_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

PYTORCH_SCRIPT = REPO_ROOT / "benchmarks" / "pytorch" / "mnist_main_fl_structured.py"
RESULTS_DIR = REPO_ROOT / "fl_results" / "v2"      # overridden by --out-dir
FIGURES_DIR = REPO_ROOT / "results" / "figures_v2"  # overridden by --fig-dir

DATA_PATH = str(REPO_ROOT / "data")

NUM_CLIENTS = 3
NUM_ROUNDS = 5
NUM_CLASSES = 10
SEED = 42

_LOCAL = {
    "batch_size": _HW.suggested_batch_size,
    "gradient_accumulation": _HW.suggested_gradient_accumulation,
    "use_amp": _HW.suggested_use_amp,
}

BASE_CONFIG = {
    "local_epochs": 1,
    "learning_rate": 1e-3,
    "local": _LOCAL,
    "num_classes": NUM_CLASSES,
    "data_path": DATA_PATH,
}


# ---------------------------------------------------------------------------
# Build a single shared MNIST dataset for partitioning + centralized baseline
# ---------------------------------------------------------------------------

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def load_mnist_train():
    """Return (full_train_dataset, all_targets_array)."""
    ds = datasets.MNIST(DATA_PATH, train=True, download=True, transform=_TRANSFORM)
    targets = np.array(ds.targets)
    return ds, targets


# ---------------------------------------------------------------------------
# Wrapper that injects client_indices into build_dataloader
# ---------------------------------------------------------------------------

def make_indexed_dataloader_factory(full_dataset):
    """Return a build_dataloader(config, split) that uses config['client_indices']."""

    def build_dataloader(config: dict, split: str = "train") -> DataLoader:
        batch_size = config.get("local", {}).get("batch_size", 16)
        idx = config.get("client_indices", None)
        if idx is None:
            # Should not happen in client mode; fall back to whole dataset
            ds = full_dataset
        else:
            ds = Subset(full_dataset, list(idx))

        if split == "train":
            return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

        # For val we use the same subset (we don't evaluate in this experiment)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return build_dataloader


# ---------------------------------------------------------------------------
# Configure FLClient with the wrapped dataloader
# ---------------------------------------------------------------------------

def make_clients(client_indices_list, full_dataset):
    """Build FLClients and replace mod.build_dataloader with our subset version."""
    clients = []
    factory = make_indexed_dataloader_factory(full_dataset)
    torch.manual_seed(SEED)
    for cid, idx in enumerate(client_indices_list):
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["client_indices"] = idx
        client = FLClient(f"site-{cid + 1}", PYTORCH_SCRIPT, cfg)
        # Override dataloader builder on this client's loaded module.
        client.mod.build_dataloader = factory
        clients.append(client)
    return clients


def run_fl_setting(setting_name, client_indices_list, full_dataset, num_rounds):
    """Run NUM_ROUNDS of FedAvg for one partitioning. Return per-round records."""
    print("\n" + "=" * 64)
    print(f"FL setting: {setting_name} | {NUM_CLIENTS} clients | {num_rounds} rounds")
    print("=" * 64)

    clients = make_clients(client_indices_list, full_dataset)

    # Server uses the first client's initial weights so all clients start equal.
    server = FLServer(
        global_weights=clients[0].get_weights(),
        config={"num_rounds": num_rounds},
        results_dir=str(RESULTS_DIR),
    )
    # Force every client weight equal at round 0
    init_w = clients[0].get_weights()
    for c in clients[1:]:
        c.set_weights(init_w)

    round_records = []
    for r in range(1, num_rounds + 1):
        print(f"\nRound {r}/{num_rounds}")
        # We need per-client losses too; capture them by running a per-client loop
        # mirroring server.run_round but storing each client's loss explicitly.
        t0 = time.time()
        all_weights = []
        all_num_samples = []
        per_client_losses = {}
        per_client_n = {}
        for client in clients:
            client.set_weights(server.weights)
            updated_w, metrics = client.local_train()
            all_weights.append(updated_w)
            all_num_samples.append(metrics["num_samples"])
            per_client_losses[client.client_id] = metrics["loss"]
            per_client_n[client.client_id] = metrics["num_samples"]
            print(f"    [{client.client_id}] loss={metrics['loss']:.4f} "
                  f"steps={metrics['steps']} n={metrics['num_samples']} "
                  f"{metrics['elapsed_sec']}s")
        # Sample-weighted FedAvg (weights AND loss weighted by n_k / n)
        server.weights = server._fedavg(all_weights, all_num_samples)
        agg_loss = server._weighted_mean(
            [per_client_losses[c.client_id] for c in clients], all_num_samples
        )
        elapsed = round(time.time() - t0, 2)
        print(f"  Aggregated loss (sample-weighted): {agg_loss:.4f}  ({elapsed}s)")
        round_records.append({
            "round": r,
            "aggregated_loss": round(agg_loss, 6),
            "aggregated_loss_uniform": round(
                sum(per_client_losses.values()) / len(per_client_losses), 6),
            "per_client_loss": per_client_losses,
            "client_num_samples": per_client_n,
            "aggregation": "fedavg_sample_weighted",
            "elapsed_sec": elapsed,
        })

    runtime = {
        "torch_device": str(clients[0].device),
        "amp_active": bool(all(c.amp_active for c in clients)),
    }
    return round_records, runtime


# ---------------------------------------------------------------------------
# Centralized baseline
# ---------------------------------------------------------------------------

def run_centralized(full_dataset, num_rounds):
    """Train a single model on the full dataset for the same total epochs.

    To match total optimizer work approximately: FL runs num_rounds rounds *
    NUM_CLIENTS clients * 1 local epoch each = NUM_CLIENTS*num_rounds epochs
    of local data across the federation, but each client only sees ~1/N of
    the data, so the centralized model's "epoch" over the FULL dataset is
    comparable to ~1 round of FL. We therefore train for `num_rounds` epochs
    over the full dataset and record per-epoch loss.
    """
    print("\n" + "=" * 64)
    print(f"Centralized baseline | full dataset | {num_rounds} epochs")
    print("=" * 64)

    # Reuse the structured module's model + train_step
    sys.path.insert(0, str(PYTORCH_SCRIPT.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("fl_central_module", str(PYTORCH_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = mod.build_model(BASE_CONFIG).to(device)
    bs = BASE_CONFIG["local"]["batch_size"]
    dl = DataLoader(full_dataset, batch_size=bs, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_CONFIG["learning_rate"])

    records = []
    for ep in range(1, num_rounds + 1):
        t0 = time.time()
        model.train()
        total_loss, n_steps = 0.0, 0
        for batch in dl:
            opt.zero_grad()
            loss = mod.train_step(model, batch, None, BASE_CONFIG)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_steps += 1
        avg_loss = total_loss / max(n_steps, 1)
        elapsed = round(time.time() - t0, 2)
        print(f"  Epoch {ep}/{num_rounds} loss={avg_loss:.4f} "
              f"steps={n_steps} {elapsed}s")
        records.append({
            "round": ep,
            "loss": round(avg_loss, 6),
            "steps": n_steps,
            "elapsed_sec": elapsed,
        })
    return records


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(results, figpath_pdf, figpath_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=False)

    panel_specs = [
        ("iid", "IID", "#2563EB"),
        ("dirichlet_0.5", r"Dirichlet $\alpha=0.5$", "#16A34A"),
        ("dirichlet_0.1", r"Dirichlet $\alpha=0.1$", "#DC2626"),
        ("centralized", "Centralized (full data)", "#7C3AED"),
    ]

    for ax, (key, title, color) in zip(axes, panel_specs):
        recs = results[key]["rounds"] if key != "centralized" else results[key]["epochs"]
        rounds = [r["round"] for r in recs]
        loss_key = "aggregated_loss" if key != "centralized" else "loss"
        agg = [r[loss_key] for r in recs]
        ax.plot(rounds, agg, marker="o", linewidth=2.2, markersize=8,
                color=color, label=("Aggregated" if key != "centralized" else "Train loss"))

        if key != "centralized":
            # Per-client traces
            client_ids = sorted(recs[0]["per_client_loss"].keys())
            for i, cid in enumerate(client_ids):
                series = [r["per_client_loss"][cid] for r in recs]
                ax.plot(rounds, series, marker=".", linewidth=1.0,
                        alpha=0.55, label=cid)

        ax.set_xlabel("Round" if key != "centralized" else "Epoch", fontsize=11)
        ax.set_ylabel("Loss (NLL)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(rounds)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(str(figpath_pdf), dpi=300, bbox_inches="tight")
    fig.savefig(str(figpath_png), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {figpath_pdf} and {figpath_png}")


# ---------------------------------------------------------------------------
# Notes writer
# ---------------------------------------------------------------------------

def write_notes(results, partition_summaries, runtime_info, notes_path):
    def fmt_pcl(rounds):
        first, last = rounds[0]["aggregated_loss"], rounds[-1]["aggregated_loss"]
        drop = first - last
        return first, last, drop

    iid_first, iid_last, iid_drop = fmt_pcl(results["iid"]["rounds"])
    a5_first, a5_last, a5_drop = fmt_pcl(results["dirichlet_0.5"]["rounds"])
    a1_first, a1_last, a1_drop = fmt_pcl(results["dirichlet_0.1"]["rounds"])
    cen_first = results["centralized"]["epochs"][0]["loss"]
    cen_last = results["centralized"]["epochs"][-1]["loss"]
    cen_drop = cen_first - cen_last

    def hist_table(summary, label):
        rows = [f"### {label}\n"]
        rows.append("| Client | Size | Class histogram (0..9) |")
        rows.append("|--------|------|------------------------|")
        for s in summary:
            rows.append(
                f"| site-{s['client_id'] + 1} | {s['size']} | {s['class_hist']} |"
            )
        return "\n".join(rows)

    iid_hist = hist_table(partition_summaries["iid"], "IID")
    a5_hist = hist_table(partition_summaries["dirichlet_0.5"], r"Dirichlet alpha=0.5")
    a1_hist = hist_table(partition_summaries["dirichlet_0.1"], r"Dirichlet alpha=0.1")

    rt = results["config"]
    device_line = (f"`{rt['torch_device']}` ({rt['torch_cuda_device_name']}; "
                   f"nvidia-smi: {rt['nvidia_smi_gpu_name']})"
                   if rt["cuda_available"] else "`cpu`")
    bs = _LOCAL["batch_size"]

    def mono(rounds, key="aggregated_loss"):
        losses = [r[key] for r in rounds]
        ups = [rounds[i]["round"] for i in range(1, len(losses)) if losses[i] >= losses[i - 1]]
        return (len(ups) == 0), ups

    def mono_text(label, rounds, key="aggregated_loss"):
        m, ups = mono(rounds, key)
        if m:
            return f"**{label}**: loss decreased monotonically over all {len(rounds)} rounds."
        return (f"**{label}**: loss did NOT decrease monotonically "
                f"(non-decrease at round(s) {', '.join(map(str, ups))}).")

    def spread(rounds):
        # mean over rounds of (max - min) per-client loss
        vals = [max(r["per_client_loss"].values()) - min(r["per_client_loss"].values())
                for r in rounds]
        return sum(vals) / len(vals)

    mono_lines = "\n".join([
        "- " + mono_text("IID", results["iid"]["rounds"]),
        "- " + mono_text("Dirichlet alpha=0.5", results["dirichlet_0.5"]["rounds"]),
        "- " + mono_text("Dirichlet alpha=0.1", results["dirichlet_0.1"]["rounds"]),
        "- " + mono_text("Centralized", results["centralized"]["epochs"], key="loss"),
    ])
    spread_lines = "\n".join(
        f"- **{lab}**: mean per-round spread (max - min per-client loss) = {spread(results[k]['rounds']):.4f}"
        for k, lab in [("iid", "IID"), ("dirichlet_0.5", "Dirichlet alpha=0.5"),
                       ("dirichlet_0.1", "Dirichlet alpha=0.1")]
    )
    per_round_rows = "\n".join(
        f"| {i + 1} | {results['iid']['rounds'][i]['aggregated_loss']:.4f} "
        f"| {results['dirichlet_0.5']['rounds'][i]['aggregated_loss']:.4f} "
        f"| {results['dirichlet_0.1']['rounds'][i]['aggregated_loss']:.4f} "
        f"| {results['centralized']['epochs'][i]['loss']:.4f} |"
        for i in range(len(results["iid"]["rounds"]))
    )

    content = f"""# AutoFL Non-IID FL Simulation Notes (v2)

Generated by `fl_runtime/run_simulation_non_iid.py`; every statement below is
computed from `simulation_non_iid_results.json`. Do not hand-edit.

## Reviewer concern addressed

A reviewer noted that IID convergence (the original Section 5 simulation) is
the easy case for FedAvg, and that the paper should also report convergence
under realistic non-IID partitions where FedAvg is known to slow down or
oscillate. This experiment adds three direct comparisons on identical
infrastructure:

1. **IID** (re-run for paired comparison)
2. **Dirichlet alpha = 0.5** (mild label skew)
3. **Dirichlet alpha = 0.1** (severe label skew; some clients dominated by 1-2 classes)
4. **Centralized baseline**: a single trainer with access to the full MNIST training set

The same FLClient/FLServer machinery, model architecture, optimizer, and
hardware-detector-driven config are used across all four settings, so any
difference in convergence is attributable to the data partition, not to
implementation drift.

## Setup

- **Script**: `fl_runtime/run_simulation_non_iid.py`
- **Partitioner**: `fl_runtime/non_iid_partition.py` (per-class Dirichlet)
- **Model**: same CNN as `benchmarks/pytorch/mnist_main_fl_structured.py`
- **Dataset**: MNIST train (60,000 examples), 10 classes
- **Clients**: {NUM_CLIENTS} (site-1 .. site-{NUM_CLIENTS})
- **Rounds**: {runtime_info['num_rounds']}
- **Local epochs / round**: 1
- **Optimizer**: AdamW, lr = 1e-3
- **Batch size (auto)**: {bs}
- **Aggregation**: sample-weighted FedAvg (w = sum_k n_k/n w_k; aggregated loss weighted the same way)
- **Seed**: {rt['seed']} (partitions + model init)
- **torch device actually used**: {device_line}
- **CUDA available**: {rt['cuda_available']}
- **AMP requested / actually active**: {rt['use_amp']} / {rt['amp_active']}
- **torch**: {rt['torch_version']}
- **Subset note**: {runtime_info.get('subset_note', 'full dataset used')}

## Partition diagnostics

{iid_hist}

{a5_hist}

{a1_hist}

## Results

| Setting | First-round loss | Final loss | Reduction |
|---------|------------------|-----------|-----------|
| IID                  | {iid_first:.4f} | {iid_last:.4f} | {iid_drop:.4f} |
| Dirichlet alpha=0.5  | {a5_first:.4f} | {a5_last:.4f} | {a5_drop:.4f} |
| Dirichlet alpha=0.1  | {a1_first:.4f} | {a1_last:.4f} | {a1_drop:.4f} |
| Centralized          | {cen_first:.4f} | {cen_last:.4f} | {cen_drop:.4f} |

### Per-round aggregated training loss (sample-weighted)

| Round | IID | Dirichlet 0.5 | Dirichlet 0.1 | Centralized (epoch) |
|-------|-----|---------------|---------------|---------------------|
{per_round_rows}

Total wall-clock time: {runtime_info['total_sec']:.1f}s.

## Computed observations

{mono_lines}

{spread_lines}

Training loss is measured on each client's own (possibly skewed) local
distribution, so a lower aggregated loss under severe skew does not imply
better generalisation; see `non_iid_accuracy_results.json` (held-out test
accuracy, overall and macro-averaged) for that.

For the AutoFL paper, the takeaway is *not* that FedAvg solves non-IID FL
(it does not — that is an algorithms research problem) but that the
**runtime produced by AutoFL's structured conversion is fully compatible
with non-IID experimentation**: switching partition strategy required only
a Dirichlet partitioner module and a one-line dataloader override. No edits
to the converted client script were needed.

## Files

- `{RESULTS_DIR.name}/simulation_non_iid_results.json` — per-round + per-client losses
- `{FIGURES_DIR.name}/fig7_non_iid.{{pdf,png}}` — 4-panel convergence figure
- `fl_runtime/run_simulation_non_iid.py` — driver
- `fl_runtime/non_iid_partition.py` — Dirichlet partitioner
"""
    notes_path.write_text(content)
    print(f"Notes saved to {notes_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    global RESULTS_DIR, FIGURES_DIR
    p = argparse.ArgumentParser(description="AutoFL non-IID FL simulation (v2)")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--fig-dir", default=str(FIGURES_DIR))
    args = p.parse_args(argv)
    RESULTS_DIR = Path(args.out_dir)
    FIGURES_DIR = Path(args.fig_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("AutoFL Non-IID FL Simulation (v2)")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")

    # Decide subset / rounds based on a coarse runtime estimate. We start with
    # the full configuration and fall back if necessary.
    full_ds, all_targets = load_mnist_train()

    # Parametric subset: if env var AUTOFL_NONIID_SUBSET is set, use it.
    subset_n = int(os.environ.get("AUTOFL_NONIID_SUBSET", "0"))
    num_rounds = int(os.environ.get("AUTOFL_NONIID_ROUNDS", str(NUM_ROUNDS)))
    subset_note = "full dataset (60k)"
    if subset_n > 0:
        rng = np.random.default_rng(SEED)
        chosen = rng.choice(len(full_ds), size=subset_n, replace=False)
        chosen.sort()
        full_ds = Subset(full_ds, chosen.tolist())
        all_targets = all_targets[chosen]
        subset_note = f"subset of {subset_n} samples (random, seed={SEED})"

    n_total = len(full_ds)
    print(f"Working on {n_total} samples, {num_rounds} rounds, {NUM_CLIENTS} clients.")

    # Partitions
    iid_idx = iid_partition(n_total, NUM_CLIENTS, seed=SEED)
    dir05_idx = dirichlet_partition(all_targets, NUM_CLIENTS, NUM_CLASSES, alpha=0.5,
                                    seed=SEED)
    dir01_idx = dirichlet_partition(all_targets, NUM_CLIENTS, NUM_CLASSES, alpha=0.1,
                                    seed=SEED)

    summaries = {
        "iid": partition_summary(iid_idx, all_targets, NUM_CLASSES),
        "dirichlet_0.5": partition_summary(dir05_idx, all_targets, NUM_CLASSES),
        "dirichlet_0.1": partition_summary(dir01_idx, all_targets, NUM_CLASSES),
    }
    for name, summ in summaries.items():
        print(f"\nPartition {name}:")
        for s in summ:
            print(f"  site-{s['client_id'] + 1}: size={s['size']} hist={s['class_hist']}")

    # Run all four settings
    iid_records, rt_iid = run_fl_setting("IID", iid_idx, full_ds, num_rounds)
    dir05_records, rt_05 = run_fl_setting("Dirichlet alpha=0.5", dir05_idx, full_ds, num_rounds)
    dir01_records, rt_01 = run_fl_setting("Dirichlet alpha=0.1", dir01_idx, full_ds, num_rounds)
    central_records = run_centralized(full_ds, num_rounds)
    cuda = torch.cuda.is_available()

    results = {
        "iid": {
            "num_clients": NUM_CLIENTS,
            "num_rounds": num_rounds,
            "partition_summary": summaries["iid"],
            "rounds": iid_records,
        },
        "dirichlet_0.5": {
            "num_clients": NUM_CLIENTS,
            "num_rounds": num_rounds,
            "alpha": 0.5,
            "partition_summary": summaries["dirichlet_0.5"],
            "rounds": dir05_records,
        },
        "dirichlet_0.1": {
            "num_clients": NUM_CLIENTS,
            "num_rounds": num_rounds,
            "alpha": 0.1,
            "partition_summary": summaries["dirichlet_0.1"],
            "rounds": dir01_records,
        },
        "centralized": {
            "num_epochs": num_rounds,
            "epochs": central_records,
        },
        "config": {
            "version": "v2",
            "aggregation": "fedavg_sample_weighted",
            "subset_note": subset_note,
            "batch_size": _LOCAL["batch_size"],
            "use_amp": _LOCAL["use_amp"],
            "amp_active": bool(rt_iid["amp_active"] and rt_05["amp_active"] and rt_01["amp_active"]),
            "torch_device": rt_iid["torch_device"],
            "cuda_available": bool(cuda),
            "torch_cuda_device_name": torch.cuda.get_device_name(0) if cuda else None,
            "nvidia_smi_gpu_name": _HW.primary_gpu_name if _HW.has_cuda else None,
            "torch_version": torch.__version__,
            "seed": SEED,
        },
    }

    out_json = RESULTS_DIR / "simulation_non_iid_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON saved to {out_json}")

    fig_pdf = FIGURES_DIR / "fig7_non_iid.pdf"
    fig_png = FIGURES_DIR / "fig7_non_iid.png"
    make_plot(results, fig_pdf, fig_png)

    total = time.time() - t_start
    runtime_info = {
        "total_sec": total,
        "num_rounds": num_rounds,
        "subset_note": subset_note,
    }

    notes_path = RESULTS_DIR / "simulation_non_iid_notes.md"
    write_notes(results, summaries, runtime_info, notes_path)

    print(f"\nTotal simulation time: {total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()

"""
AutoFL end-to-end FL simulation (v2: partitioned IID + sample-weighted FedAvg).

Runs two simulations:
  1. PyTorch MNIST: 3 clients, 5 rounds
  2. Lightning GAN:  2 clients, 3 rounds

Each client trains on a DISJOINT, equal-size IID shard of the training split
returned by the generated module's build_dataloader (a seeded random
permutation split with np.array_split, seed=42). Aggregation is
sample-weighted FedAvg (fl_runtime/server.py).

Outputs (defaults; v1 archives in fl_results/ are never overwritten):
  <out-dir>/simulation_results.json      (default fl_results/v2/)
  <out-dir>/simulation_notes.md          (generated from the data, never hand-edited)
  <fig-dir>/fig5_fl_convergence.{pdf,png} (default results/figures_v2/)
"""
import argparse
import os
import sys
import json
import time
import copy
from pathlib import Path

import numpy as np

# Make fl_runtime importable (must happen before hardware import)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fl_runtime"))

from hardware.detector import detect, print_profile
from fl_runtime.client import FLClient
from fl_runtime.server import FLServer

# ---------------------------------------------------------------------------
# Hardware detection — sets CUDA visibility and adapts training params
# ---------------------------------------------------------------------------
_HW = detect()
print_profile(_HW)

if not _HW.has_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PYTORCH_SCRIPT = REPO_ROOT / "benchmarks" / "pytorch" / "mnist_main_fl_structured.py"
LIGHTNING_SCRIPT = REPO_ROOT / "benchmarks" / "lightning" / "mnist_lite_fl_structured.py"
DATA_PATH = str(REPO_ROOT / "data")

# ---------------------------------------------------------------------------
# Config — adapted from hardware profile
# ---------------------------------------------------------------------------
SEED = 42

_LOCAL = {
    "batch_size": _HW.suggested_batch_size,
    "gradient_accumulation": _HW.suggested_gradient_accumulation,
    "use_amp": _HW.suggested_use_amp,
}

PYTORCH_CONFIG = {
    "local_epochs": 1,
    "learning_rate": 1e-3,
    "local": _LOCAL,
    "num_classes": 10,
    "data_path": DATA_PATH,
    "seed": SEED,
}

LIGHTNING_CONFIG = {
    "local_epochs": 1,
    "learning_rate": 1e-3,
    "local": _LOCAL,
    "num_classes": 10,
    "data_path": DATA_PATH,
    "seed": SEED,
}

NUM_PYTORCH_ROUNDS = 5
NUM_LIGHTNING_ROUNDS = 3
PYTORCH_CLIENT_IDS = ["site-1", "site-2", "site-3"]
LIGHTNING_CLIENT_IDS = ["lgn-1", "lgn-2"]


# ---------------------------------------------------------------------------
# IID partitioning: disjoint equal shards of the module's own train split
# ---------------------------------------------------------------------------

def iid_shards(num_samples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    """Seeded random permutation split into `num_clients` near-equal shards
    (same approach as fl_runtime/non_iid_partition.iid_partition)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_samples)
    return [np.sort(chunk) for chunk in np.array_split(idx, num_clients)]


def shard_client_dataloader(client: FLClient, shard_rank: int, num_clients: int, seed: int):
    """Wrap client.mod.build_dataloader so that the 'train' split is restricted
    to shard `shard_rank` of a disjoint equal split (the 'val' split is passed
    through unchanged). Returns a callable that reports the shard size after the
    first call."""
    orig_build = client.mod.build_dataloader
    info = {"shard_size": None, "full_size": None}

    def build_dataloader(config: dict, split: str = "train") -> DataLoader:
        dl = orig_build(config, split=split)
        if split != "train":
            return dl
        ds = dl.dataset
        shards = iid_shards(len(ds), num_clients, seed)
        my = shards[shard_rank]
        info["shard_size"] = int(len(my))
        info["full_size"] = int(len(ds))
        return DataLoader(
            Subset(ds, my.tolist()),
            batch_size=dl.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=bool(getattr(dl, "drop_last", False)),
        )

    client.mod.build_dataloader = build_dataloader
    return info


def make_sharded_clients(script: Path, config: dict, client_ids: list[str], seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    clients = [FLClient(cid, script, copy.deepcopy(config)) for cid in client_ids]
    shard_infos = []
    for rank, client in enumerate(clients):
        shard_infos.append(shard_client_dataloader(client, rank, len(clients), seed))
    return clients, shard_infos


# ---------------------------------------------------------------------------
# Preflight helper
# ---------------------------------------------------------------------------

def run_preflight(client: FLClient) -> dict:
    """Quick sanity check: build dataloader, do one forward pass."""
    try:
        dl = client.mod.build_dataloader(client.config, split="train")
        batch = next(iter(dl))
        client.model.eval()
        with torch.no_grad():
            loss = client.mod.train_step(client.model, batch, None, client.config)
        return {"success": True, "message": f"loss={loss.item():.4f}"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ---------------------------------------------------------------------------
# Generic simulation
# ---------------------------------------------------------------------------

def run_simulation(title, script, config, client_ids, num_rounds, out_dir, seed):
    print("\n" + "=" * 60)
    print(f"{title} | {len(client_ids)} clients | {num_rounds} rounds | seed={seed}")
    print("=" * 60)

    clients, shard_infos = make_sharded_clients(script, config, client_ids, seed)
    # All clients start from identical weights
    init_w = clients[0].get_weights()
    for c in clients[1:]:
        c.set_weights(init_w)

    print("\nRunning preflight checks ...")
    server = FLServer(
        global_weights=init_w,
        config={"num_rounds": num_rounds},
        results_dir=str(out_dir),
    )
    for client in clients:
        result = run_preflight(client)
        server.register_preflight(client.client_id, result)

    ready = server.ready_clients()
    print(f"  {len(ready)}/{len(clients)} clients ready: {ready}")
    shard_sizes = {c.client_id: si["shard_size"] for c, si in zip(clients, shard_infos)}
    full_size = shard_infos[0]["full_size"]
    print(f"  IID shards (disjoint, of {full_size} train examples): {shard_sizes}")

    print(f"\nStarting FL: {num_rounds} rounds, {len(ready)} clients")
    print("-" * 60)
    round_records = []
    active = [c for c in clients if c.client_id in ready]

    for r in range(1, num_rounds + 1):
        print(f"\nRound {r}/{num_rounds}")
        rr = server.run_round(active, r)
        print(f"  Aggregated loss: {rr.aggregated_loss:.4f}  ({rr.elapsed_sec}s)")
        round_records.append({
            "round": r,
            "aggregated_loss": rr.aggregated_loss,
            "per_client_loss": {cid: v["loss"] for cid, v in rr.extra["per_client"].items()},
            "client_num_samples": rr.extra["client_num_samples"],
            "aggregation": rr.extra["aggregation"],
            "elapsed_sec": rr.elapsed_sec,
            "clients": rr.participating_clients,
        })

    print(f"\n--- Convergence Table ({title}) ---")
    print(f"{'Round':>6}  {'Agg. Loss':>10}  {'Elapsed (s)':>12}")
    print("-" * 34)
    for rec in round_records:
        print(f"{rec['round']:>6}  {rec['aggregated_loss']:>10.4f}  {rec['elapsed_sec']:>12.2f}")

    runtime = runtime_info(active)
    partition = {
        "type": "iid_disjoint_equal_shards",
        "seed": seed,
        "train_split_size": full_size,
        "shard_sizes": shard_sizes,
    }
    return round_records, partition, runtime


def runtime_info(clients) -> dict:
    dev = clients[0].device
    cuda = torch.cuda.is_available()
    return {
        "torch_device": str(dev),
        "cuda_available": bool(cuda),
        "torch_cuda_device_name": torch.cuda.get_device_name(0) if cuda else None,
        "nvidia_smi_gpu_name": _HW.primary_gpu_name if _HW.has_cuda else None,
        "amp_requested": bool(_LOCAL["use_amp"]),
        "amp_active": bool(all(c.amp_active for c in clients)),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "batch_size": _LOCAL["batch_size"],
        "gradient_accumulation": _LOCAL["gradient_accumulation"],
        "cpu_count": _HW.cpu_count,
        "ram_gb": _HW.ram_gb,
    }


# ---------------------------------------------------------------------------
# Save JSON results
# ---------------------------------------------------------------------------

def save_results(pt, lgn, out_dir):
    pt_records, pt_partition, pt_runtime = pt
    lgn_records, lgn_partition, lgn_runtime = lgn
    out = {
        "version": "v2",
        "seed": SEED,
        "aggregation": "fedavg_sample_weighted",
        "pytorch_mnist": {
            "framework": "PyTorch",
            "script": str(PYTORCH_SCRIPT),
            "num_clients": len(PYTORCH_CLIENT_IDS),
            "num_rounds": NUM_PYTORCH_ROUNDS,
            "config": PYTORCH_CONFIG,
            "partition": pt_partition,
            "runtime": pt_runtime,
            "rounds": pt_records,
        },
        "lightning_gan": {
            "framework": "Lightning",
            "script": str(LIGHTNING_SCRIPT),
            "num_clients": len(LIGHTNING_CLIENT_IDS),
            "num_rounds": NUM_LIGHTNING_ROUNDS,
            "config": LIGHTNING_CONFIG,
            "partition": lgn_partition,
            "runtime": lgn_runtime,
            "rounds": lgn_records,
        },
    }
    path = out_dir / "simulation_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")
    return out


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def make_convergence_plot(pytorch_records, lightning_records, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    rounds_pt = [r["round"] for r in pytorch_records]
    losses_pt = [r["aggregated_loss"] for r in pytorch_records]
    ax = axes[0]
    ax.plot(rounds_pt, losses_pt, marker="o", linewidth=2, markersize=7,
            color="#2563EB", label=f"FedAvg ({len(PYTORCH_CLIENT_IDS)} clients, disjoint IID shards)")
    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Aggregated Loss (NLL)", fontsize=12)
    ax.set_title("PyTorch MNIST — FL Convergence", fontsize=13)
    ax.set_xticks(rounds_pt)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    rounds_lgn = [r["round"] for r in lightning_records]
    losses_lgn = [r["aggregated_loss"] for r in lightning_records]
    ax2 = axes[1]
    ax2.plot(rounds_lgn, losses_lgn, marker="s", linewidth=2, markersize=7,
             color="#DC2626", label=f"FedAvg ({len(LIGHTNING_CLIENT_IDS)} clients, disjoint IID shards)")
    ax2.set_xlabel("FL Round", fontsize=12)
    ax2.set_ylabel("Aggregated Loss (GAN)", fontsize=12)
    ax2.set_title("Lightning GAN — FL Convergence", fontsize=13)
    ax2.set_xticks(rounds_lgn)
    ax2.tick_params(labelsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    pdf_path = fig_dir / "fig5_fl_convergence.pdf"
    png_path = fig_dir / "fig5_fl_convergence.png"
    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot saved to {pdf_path} and {png_path}")


# ---------------------------------------------------------------------------
# Notes (every claim computed from the recorded data)
# ---------------------------------------------------------------------------

def monotonic_summary(records) -> tuple[bool, list[int]]:
    """Return (strictly_decreasing, rounds_where_loss_increased)."""
    losses = [r["aggregated_loss"] for r in records]
    ups = [records[i]["round"] for i in range(1, len(losses)) if losses[i] >= losses[i - 1]]
    return (len(ups) == 0), ups


def describe_monotonicity(name, records) -> str:
    mono, ups = monotonic_summary(records)
    if mono:
        return f"{name}: aggregated loss decreased monotonically over all {len(records)} rounds."
    return (f"{name}: aggregated loss did NOT decrease monotonically "
            f"(non-decrease at round(s) {', '.join(map(str, ups))}).")


def save_notes(pt, lgn, out_dir, total_wall_sec):
    pt_records, pt_partition, pt_runtime = pt
    lgn_records, lgn_partition, lgn_runtime = lgn

    def stats(records):
        first = records[0]["aggregated_loss"]
        last = records[-1]["aggregated_loss"]
        drop = first - last
        pct = 100.0 * drop / first if first > 0 else 0.0
        tot = sum(r["elapsed_sec"] for r in records)
        return first, last, drop, pct, tot

    pt_first, pt_last, pt_drop, pt_pct, pt_total_time = stats(pt_records)
    lgn_first, lgn_last, lgn_drop, lgn_pct, lgn_total_time = stats(lgn_records)

    def table(records):
        return "\n".join(
            f"| {r['round']} | {r['aggregated_loss']:.4f} | "
            + " | ".join(f"{r['per_client_loss'][c]:.4f}" for c in r['clients'])
            + f" | {r['elapsed_sec']:.2f} |"
            for r in records
        )

    def header(records):
        cids = records[0]["clients"]
        return ("| Round | Agg. loss (sample-weighted) | " + " | ".join(cids) + " | Elapsed (s) |\n"
                "|-------|-----------------------------|" + "|".join("-" * (len(c) + 2) for c in cids) + "|-------------|")

    rt = pt_runtime  # both simulations ran in the same process/device
    device_line = (f"{rt['torch_device']} "
                   f"({rt['torch_cuda_device_name']}; nvidia-smi: {rt['nvidia_smi_gpu_name']})"
                   if rt["cuda_available"] else "cpu")
    pt_mono_line = describe_monotonicity("PyTorch MNIST", pt_records)
    lgn_mono_line = describe_monotonicity("Lightning GAN", lgn_records)
    pt_mono, _ = monotonic_summary(pt_records)
    lgn_mono, _ = monotonic_summary(lgn_records)

    content = f"""# AutoFL FL Simulation Notes (v2)

Generated by `fl_runtime/run_simulation.py`; every statement below is computed
from `simulation_results.json` in this directory. Do not hand-edit.

## Runtime actually used

- **torch device**: `{rt['torch_device']}`
- **CUDA available**: {rt['cuda_available']}
- **GPU (torch)**: {rt['torch_cuda_device_name']}
- **GPU (nvidia-smi)**: {rt['nvidia_smi_gpu_name']}
- **AMP requested (hardware detector)**: {rt['amp_requested']}
- **AMP actually active** (`use_amp and device.type == "cuda"`): {rt['amp_active']}
- **torch**: {rt['torch_version']}  |  **python**: {rt['python_version']}
- **CPU cores**: {rt['cpu_count']}  |  **RAM**: {rt['ram_gb']} GB
- **batch_size** (auto): {rt['batch_size']}  |  **gradient_accumulation** (auto): {rt['gradient_accumulation']}
- **seed**: {SEED} (torch.manual_seed / np.random.seed / shard permutation)
- **Aggregation**: sample-weighted FedAvg, w = sum_k (n_k / n) w_k

## Setup

### Simulation 1 — PyTorch MNIST
- **Script**: `benchmarks/pytorch/mnist_main_fl_structured.py`
- **Clients**: {len(PYTORCH_CLIENT_IDS)} ({', '.join(PYTORCH_CLIENT_IDS)})
- **Partition**: disjoint equal IID shards of the module's {pt_partition['train_split_size']}-example train split: {pt_partition['shard_sizes']}
- **Rounds**: {NUM_PYTORCH_ROUNDS}; **local_epochs**: 1; **lr**: 1e-3

### Simulation 2 — Lightning GAN (mnist_lite)
- **Script**: `benchmarks/lightning/mnist_lite_fl_structured.py`
- **Clients**: {len(LIGHTNING_CLIENT_IDS)} ({', '.join(LIGHTNING_CLIENT_IDS)})
- **Partition**: disjoint equal IID shards of the module's {lgn_partition['train_split_size']}-example train split: {lgn_partition['shard_sizes']}
- **Rounds**: {NUM_LIGHTNING_ROUNDS}; **local_epochs**: 1; **lr**: 1e-3

---

## Convergence Results

### PyTorch MNIST ({len(PYTORCH_CLIENT_IDS)} clients, {NUM_PYTORCH_ROUNDS} rounds)

{header(pt_records)}
{table(pt_records)}

- Initial loss: {pt_first:.4f}
- Final loss:   {pt_last:.4f}
- Reduction:    {pt_drop:.4f} ({pt_pct:.1f}%)
- Monotonic:    {pt_mono}
- Total time:   {pt_total_time:.1f}s

### Lightning GAN ({len(LIGHTNING_CLIENT_IDS)} clients, {NUM_LIGHTNING_ROUNDS} rounds)

{header(lgn_records)}
{table(lgn_records)}

- Initial loss: {lgn_first:.4f}
- Final loss:   {lgn_last:.4f}
- Reduction:    {lgn_drop:.4f} ({lgn_pct:.1f}%)
- Monotonic:    {lgn_mono}
- Total time:   {lgn_total_time:.1f}s

Total wall-clock for both simulations (incl. preflight, data loading): {total_wall_sec:.1f}s

---

## Computed statements for the paper

- {pt_mono_line}
- {lgn_mono_line}
- The PyTorch simulation reduced the aggregated loss by {pt_pct:.1f}% over
  {NUM_PYTORCH_ROUNDS} rounds ({pt_total_time:.0f}s of round time on {device_line},
  AMP active: {rt['amp_active']}).
- The Lightning GAN simulation changed the aggregated loss by {lgn_pct:.1f}% over
  {NUM_LIGHTNING_ROUNDS} rounds ({lgn_total_time:.0f}s).
- Both structured-converted clients ran unmodified under FLClient/FLServer with
  disjoint IID shards and sample-weighted FedAvg.
"""
    path = out_dir / "simulation_notes.md"
    with open(path, "w") as f:
        f.write(content)
    print(f"Simulation notes saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    global SEED
    p = argparse.ArgumentParser(description="AutoFL end-to-end FL simulation (v2)")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "fl_results" / "v2"))
    p.add_argument("--fig-dir", default=str(REPO_ROOT / "results" / "figures_v2"))
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    SEED = args.seed
    PYTORCH_CONFIG["seed"] = SEED
    LIGHTNING_CONFIG["seed"] = SEED

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("AutoFL End-to-End FL Simulation (v2)")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")

    pt = run_simulation("Simulation 1: PyTorch MNIST", PYTORCH_SCRIPT, PYTORCH_CONFIG,
                        PYTORCH_CLIENT_IDS, NUM_PYTORCH_ROUNDS, out_dir, SEED)
    lgn = run_simulation("Simulation 2: Lightning GAN", LIGHTNING_SCRIPT, LIGHTNING_CONFIG,
                         LIGHTNING_CLIENT_IDS, NUM_LIGHTNING_ROUNDS, out_dir, SEED)

    total = time.time() - t_start
    save_results(pt, lgn, out_dir)
    make_convergence_plot(pt[0], lgn[0], fig_dir)
    save_notes(pt, lgn, out_dir, total)

    print(f"\nTotal simulation time: {total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()

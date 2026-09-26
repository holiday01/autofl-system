"""
Compute per-round test accuracy for the non-IID FL simulation (v2).

Re-runs the MNIST FL simulation (3 clients, 5 rounds, IID / Dirichlet 0.5 / 0.1)
with SAMPLE-WEIGHTED FedAvg and evaluates the global model on the held-out
10k MNIST test set after each round.

Two accuracy metrics are stored per round:
  accuracy_overall — micro accuracy: (#correct) / (#test examples)
  accuracy_macro   — macro-averaged per-class accuracy: mean over the 10 classes
                     of per-class recall (correct_c / total_c)
`test_accuracy` is kept as an alias of accuracy_overall for backwards
compatibility with the v1 JSON layout.

Outputs (defaults; v1 archives in fl_results/ are never overwritten):
  <out-dir>/non_iid_accuracy_results.json   (default fl_results/v2/)
  <out-dir>/non_iid_accuracy_notes.md       (generated from data)
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "fl_runtime"))

from hardware.detector import detect
from fl_runtime.client import FLClient
from fl_runtime.server import FLServer
from fl_runtime.non_iid_partition import dirichlet_partition, iid_partition, partition_summary

_HW = detect()
if not _HW.has_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

PYTORCH_SCRIPT = REPO_ROOT / "benchmarks" / "pytorch" / "mnist_main_fl_structured.py"
RESULTS_DIR = REPO_ROOT / "fl_results" / "v2"   # overridden by --out-dir

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

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def load_mnist():
    train_ds = datasets.MNIST(DATA_PATH, train=True, download=True, transform=_TRANSFORM)
    test_ds = datasets.MNIST(DATA_PATH, train=False, download=True, transform=_TRANSFORM)
    train_targets = np.array(train_ds.targets)
    return train_ds, test_ds, train_targets


def load_module():
    spec = importlib.util.spec_from_file_location("fl_mod", str(PYTORCH_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(model, test_ds, device, batch_size=256, num_classes=NUM_CLASSES) -> dict:
    """Evaluate model on the test dataset.

    Returns dict with
      accuracy_overall : micro accuracy, correct / total
      accuracy_macro   : mean over classes of per-class recall
      per_class_accuracy : list of per-class recall (len num_classes)
      per_class_support  : list of per-class test counts
    """
    model.eval()
    dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    correct_c = torch.zeros(num_classes, dtype=torch.long)
    total_c = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for data, target in dl:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            hit = pred.eq(target)
            total_c += torch.bincount(target.cpu(), minlength=num_classes)
            correct_c += torch.bincount(target.cpu()[hit.cpu()], minlength=num_classes)
    correct = int(correct_c.sum())
    total = int(total_c.sum())
    per_class = (correct_c.double() / total_c.clamp(min=1).double()).tolist()
    present = total_c > 0
    macro = float(torch.tensor(per_class)[present].mean()) if present.any() else 0.0
    return {
        "accuracy_overall": correct / total,
        "accuracy_macro": macro,
        "per_class_accuracy": [round(x, 6) for x in per_class],
        "per_class_support": total_c.tolist(),
    }


def make_indexed_dataloader_factory(full_dataset):
    def build_dataloader(config: dict, split: str = "train") -> DataLoader:
        batch_size = config.get("local", {}).get("batch_size", 16)
        idx = config.get("client_indices", None)
        if idx is None:
            ds = full_dataset
        else:
            ds = Subset(full_dataset, list(idx))
        if split == "train":
            return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return build_dataloader


def make_clients(client_indices_list, full_dataset):
    factory = make_indexed_dataloader_factory(full_dataset)
    clients = []
    torch.manual_seed(SEED)
    for cid, idx in enumerate(client_indices_list):
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["client_indices"] = idx
        client = FLClient(f"site-{cid + 1}", PYTORCH_SCRIPT, cfg)
        client.mod.build_dataloader = factory
        clients.append(client)
    return clients


def run_setting(setting_name, client_indices_list, train_ds, test_ds, num_rounds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clients = make_clients(client_indices_list, train_ds)
    init_w = clients[0].get_weights()
    for c in clients[1:]:
        c.set_weights(init_w)

    server = FLServer(
        global_weights=init_w,
        config={"num_rounds": num_rounds},
        results_dir=str(RESULTS_DIR),
    )

    mod = load_module()
    global_model = mod.build_model(BASE_CONFIG).to(device)

    def set_model_weights(model, weights):
        model.load_state_dict({k: v.clone() for k, v in weights.items()})

    set_model_weights(global_model, server.weights)
    ev0 = evaluate(global_model, test_ds, device)
    print(f"\n[{setting_name}] Round 0 (init) acc_overall={ev0['accuracy_overall']:.4f} "
          f"acc_macro={ev0['accuracy_macro']:.4f}")

    round_records = [{
        "round": 0,
        "aggregated_loss": None,
        "test_accuracy": round(ev0["accuracy_overall"], 6),
        "accuracy_overall": round(ev0["accuracy_overall"], 6),
        "accuracy_macro": round(ev0["accuracy_macro"], 6),
        "per_class_accuracy": ev0["per_class_accuracy"],
        "elapsed_sec": 0.0,
    }]
    for r in range(1, num_rounds + 1):
        t0 = time.time()
        all_weights = []
        all_n = []
        per_client_losses = {}
        per_client_n = {}
        for client in clients:
            client.set_weights(server.weights)
            updated_w, metrics = client.local_train()
            all_weights.append(updated_w)
            all_n.append(metrics["num_samples"])
            per_client_losses[client.client_id] = metrics["loss"]
            per_client_n[client.client_id] = metrics["num_samples"]
        # Sample-weighted FedAvg for both weights and the reported loss
        server.weights = server._fedavg(all_weights, all_n)
        agg_loss = server._weighted_mean(
            [per_client_losses[c.client_id] for c in clients], all_n)
        elapsed = round(time.time() - t0, 2)

        set_model_weights(global_model, server.weights)
        ev = evaluate(global_model, test_ds, device)
        print(f"  [{setting_name}] Round {r} loss={agg_loss:.4f} "
              f"acc_overall={ev['accuracy_overall']:.4f} acc_macro={ev['accuracy_macro']:.4f} "
              f"n={per_client_n} ({elapsed}s)")
        round_records.append({
            "round": r,
            "aggregated_loss": round(agg_loss, 6),
            "aggregated_loss_uniform": round(
                sum(per_client_losses.values()) / len(per_client_losses), 6),
            "per_client_loss": per_client_losses,
            "client_num_samples": per_client_n,
            "test_accuracy": round(ev["accuracy_overall"], 6),   # alias (v1 layout)
            "accuracy_overall": round(ev["accuracy_overall"], 6),
            "accuracy_macro": round(ev["accuracy_macro"], 6),
            "per_class_accuracy": ev["per_class_accuracy"],
            "elapsed_sec": elapsed,
        })
    runtime = {
        "torch_device": str(clients[0].device),
        "amp_active": bool(all(c.amp_active for c in clients)),
    }
    return round_records, runtime


def main(argv=None):
    global RESULTS_DIR
    p = argparse.ArgumentParser(description="Non-IID per-round test accuracy (v2)")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = p.parse_args(argv)
    RESULTS_DIR = Path(args.out_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    t_start = time.time()

    print("Loading MNIST...")
    train_ds, test_ds, train_targets = load_mnist()
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")

    settings = {}
    partitions = {}
    runtimes = {}

    iid_indices = iid_partition(len(train_targets), NUM_CLIENTS, seed=SEED)
    print(f"\nIID partition sizes: {[len(x) for x in iid_indices]}")
    partitions["IID"] = partition_summary(iid_indices, train_targets, NUM_CLASSES)
    settings["IID"], runtimes["IID"] = run_setting("IID", iid_indices, train_ds, test_ds, NUM_ROUNDS)

    dir05_indices = dirichlet_partition(
        train_targets, NUM_CLIENTS, num_classes=NUM_CLASSES, alpha=0.5, seed=SEED)
    print(f"\nDirichlet 0.5 partition sizes: {[len(x) for x in dir05_indices]}")
    partitions["Dirichlet_0.5"] = partition_summary(dir05_indices, train_targets, NUM_CLASSES)
    settings["Dirichlet_0.5"], runtimes["Dirichlet_0.5"] = run_setting(
        "Dir_0.5", dir05_indices, train_ds, test_ds, NUM_ROUNDS)

    dir01_indices = dirichlet_partition(
        train_targets, NUM_CLIENTS, num_classes=NUM_CLASSES, alpha=0.1, seed=SEED)
    print(f"\nDirichlet 0.1 partition sizes: {[len(x) for x in dir01_indices]}")
    partitions["Dirichlet_0.1"] = partition_summary(dir01_indices, train_targets, NUM_CLASSES)
    settings["Dirichlet_0.1"], runtimes["Dirichlet_0.1"] = run_setting(
        "Dir_0.1", dir01_indices, train_ds, test_ds, NUM_ROUNDS)

    total_sec = time.time() - t_start
    cuda = torch.cuda.is_available()
    out = {
        **settings,
        "_meta": {
            "version": "v2",
            "aggregation": "fedavg_sample_weighted",
            "seed": SEED,
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "metrics": {
                "accuracy_overall": "micro accuracy = correct / total on the 10k MNIST test set",
                "accuracy_macro": "macro-averaged per-class accuracy = mean over 10 classes of per-class recall",
                "test_accuracy": "alias of accuracy_overall (v1 layout)",
            },
            "partitions": partitions,
            "torch_device": runtimes["IID"]["torch_device"],
            "cuda_available": bool(cuda),
            "torch_cuda_device_name": torch.cuda.get_device_name(0) if cuda else None,
            "nvidia_smi_gpu_name": _HW.primary_gpu_name if _HW.has_cuda else None,
            "use_amp_requested": bool(_LOCAL["use_amp"]),
            "amp_active": bool(all(r["amp_active"] for r in runtimes.values())),
            "batch_size": _LOCAL["batch_size"],
            "torch_version": torch.__version__,
            "total_wall_sec": round(total_sec, 2),
        },
    }
    out_path = RESULTS_DIR / "non_iid_accuracy_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary tables
    print("\n=== Per-Round Test Accuracy (overall / macro) ===")
    print(f"{'Round':<8}" + "".join(f"{name:<26}" for name in settings))
    for r in range(NUM_ROUNDS + 1):
        print(f"R{r:<7}" + "".join(
            f"{recs[r]['accuracy_overall']*100:5.2f}% / {recs[r]['accuracy_macro']*100:5.2f}%      "
            for recs in settings.values()))

    # Notes (all statements computed)
    notes_path = RESULTS_DIR / "non_iid_accuracy_notes.md"
    meta = out["_meta"]
    labels = {"IID": "IID", "Dirichlet_0.5": "Dirichlet alpha=0.5", "Dirichlet_0.1": "Dirichlet alpha=0.1"}

    def row(name):
        recs = settings[name]
        r1, r5 = recs[1], recs[-1]
        return (f"| {labels[name]} | {r1['accuracy_overall']*100:.2f}% | {r5['accuracy_overall']*100:.2f}% "
                f"| {(r5['accuracy_overall']-r1['accuracy_overall'])*100:+.2f}pp "
                f"| {r1['accuracy_macro']*100:.2f}% | {r5['accuracy_macro']*100:.2f}% "
                f"| {(r5['accuracy_macro']-r1['accuracy_macro'])*100:+.2f}pp |")

    final_over = {n: settings[n][-1]["accuracy_overall"] for n in settings}
    final_macro = {n: settings[n][-1]["accuracy_macro"] for n in settings}
    above90 = [labels[n] for n, v in final_over.items() if v > 0.90]
    below90 = [labels[n] for n, v in final_over.items() if v <= 0.90]
    gap_over = (final_over["IID"] - final_over["Dirichlet_0.1"]) * 100
    gap_macro = (final_macro["IID"] - final_macro["Dirichlet_0.1"]) * 100

    def mono_acc(name, key):
        vals = [r[key] for r in settings[name][1:]]
        ups = [settings[name][i + 1]["round"] for i in range(1, len(vals)) if vals[i] < vals[i - 1]]
        return (len(ups) == 0), ups

    mono_lines = []
    for n in settings:
        m, ups = mono_acc(n, "accuracy_overall")
        mono_lines.append(f"- **{labels[n]}** overall accuracy "
                          + ("increased monotonically over rounds 1-5."
                             if m else f"decreased at round(s) {', '.join(map(str, ups))}."))

    per_round_rows = "\n".join(
        f"| {r} | " + " | ".join(
            f"{settings[n][r]['accuracy_overall']*100:.2f}% / {settings[n][r]['accuracy_macro']*100:.2f}%"
            for n in settings) + " |"
        for r in range(0, NUM_ROUNDS + 1)
    )
    loss_rows = "\n".join(
        f"| {r} | " + " | ".join(f"{settings[n][r]['aggregated_loss']:.4f}" for n in settings) + " |"
        for r in range(1, NUM_ROUNDS + 1)
    )

    device_line = (f"`{meta['torch_device']}` ({meta['torch_cuda_device_name']}; "
                   f"nvidia-smi: {meta['nvidia_smi_gpu_name']})" if meta["cuda_available"] else "`cpu`")

    with open(notes_path, "w") as f:
        f.write("# Non-IID FL Per-Round Test Accuracy (v2)\n\n")
        f.write("Generated by `eval/non_iid_accuracy.py`; every statement is computed from "
                "`non_iid_accuracy_results.json`. Do not hand-edit.\n\n")
        f.write(f"**Dataset**: MNIST — 60k train, 10k test, {NUM_CLIENTS} FL clients, {NUM_ROUNDS} rounds\n")
        f.write(f"**Aggregation**: sample-weighted FedAvg (n_k / n)\n")
        f.write(f"**Seed**: {SEED}\n")
        f.write(f"**torch device actually used**: {device_line}\n")
        f.write(f"**AMP requested / actually active**: {meta['use_amp_requested']} / {meta['amp_active']}\n")
        f.write(f"**Wall-clock (all three settings)**: {meta['total_wall_sec']:.1f}s\n\n")
        f.write("## Metrics\n\n")
        f.write("- `accuracy_overall`: micro accuracy = #correct / #test examples (10,000).\n")
        f.write("- `accuracy_macro`: macro-averaged per-class accuracy = mean over the 10 classes of per-class recall.\n\n")
        f.write("## Summary (round 1 -> round 5)\n\n")
        f.write("| Setting | R1 overall | R5 overall | gain | R1 macro | R5 macro | gain |\n")
        f.write("|---------|-----------|-----------|------|----------|----------|------|\n")
        for n in settings:
            f.write(row(n) + "\n")
        f.write("\n## Per-round test accuracy (overall / macro)\n\n")
        f.write("| Round | " + " | ".join(labels[n] for n in settings) + " |\n")
        f.write("|-------|" + "|".join("-" * (len(labels[n]) + 2) for n in settings) + "|\n")
        f.write(per_round_rows + "\n")
        f.write("\n## Per-round aggregated training loss (sample-weighted)\n\n")
        f.write("| Round | " + " | ".join(labels[n] for n in settings) + " |\n")
        f.write("|-------|" + "|".join("-" * (len(labels[n]) + 2) for n in settings) + "|\n")
        f.write(loss_rows + "\n")
        f.write("\n## Computed observations\n\n")
        f.write("\n".join(mono_lines) + "\n")
        f.write(f"- Settings with round-5 overall accuracy > 90%: {', '.join(above90) if above90 else 'none'}"
                f"{'; <= 90%: ' + ', '.join(below90) if below90 else ''}.\n")
        f.write(f"- IID minus Dirichlet-0.1 gap at round 5: {gap_over:.2f} pp (overall), "
                f"{gap_macro:.2f} pp (macro).\n")
        f.write("\n## Raw results\n\n")
        for name, recs in settings.items():
            f.write(f"\n### {name}\n\n")
            for rec in recs:
                loss_s = f"{rec['aggregated_loss']:.6f}" if rec["aggregated_loss"] is not None else "n/a"
                f.write(f"Round {rec['round']}: loss={loss_s}, "
                        f"acc_overall={rec['accuracy_overall']*100:.2f}%, "
                        f"acc_macro={rec['accuracy_macro']*100:.2f}%, elapsed={rec['elapsed_sec']}s\n")

    print(f"Notes saved: {notes_path}")


if __name__ == "__main__":
    main()

"""
Print paste-ready markdown tables comparing v1 (fl_results/*.json) and v2
(fl_results/v2/*.json) FL simulation outputs.

Usage: python eval/summarize_simulation_versions.py [--v1-dir fl_results] [--v2-dir fl_results/v2]
"""
import argparse
import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", default="fl_results")
    ap.add_argument("--v2-dir", default="fl_results/v2")
    a = ap.parse_args()
    v1d, v2d = Path(a.v1_dir), Path(a.v2_dir)

    # ---------------- simulation_results (IID PyTorch + Lightning) ----------------
    s1, s2 = load(v1d / "simulation_results.json"), load(v2d / "simulation_results.json")
    if s2:
        rt = s2["pytorch_mnist"]["runtime"]
        print("## run_simulation: runtime actually used (v2)\n")
        for k, v in rt.items():
            print(f"- {k}: {v}")
        print(f"- seed: {s2['seed']}; aggregation: {s2['aggregation']}")
        for key, title in [("pytorch_mnist", "PyTorch MNIST"), ("lightning_gan", "Lightning GAN")]:
            r1 = {r["round"]: r for r in s1[key]["rounds"]} if s1 else {}
            r2 = {r["round"]: r for r in s2[key]["rounds"]}
            part = s2[key]["partition"]
            print(f"\n### {title} ({s2[key]['num_clients']} clients, {s2[key]['num_rounds']} rounds)\n")
            print(f"v2 partition: {part['type']}, train split {part['train_split_size']}, shards {part['shard_sizes']}, seed {part['seed']}")
            print(f"v1 partition: none (each client trained on the full {part['train_split_size']}-example split); v1 aggregation: uniform mean\n")
            cids = s2[key]["rounds"][0]["clients"]
            print("| Round | v1 agg. loss | v1 elapsed (s) | v2 agg. loss (weighted) | " + " | ".join(f"v2 {c}" for c in cids) + " | v2 elapsed (s) |")
            print("|---|---|---|---|" + "---|" * len(cids) + "---|")
            for r in sorted(r2):
                a1 = r1.get(r, {})
                b = r2[r]
                print(f"| {r} | {fmt(a1.get('aggregated_loss'))} | {a1.get('elapsed_sec', 'n/a')} | {fmt(b['aggregated_loss'])} | "
                      + " | ".join(fmt(b['per_client_loss'][c]) for c in cids) + f" | {b['elapsed_sec']} |")
            l2 = [r2[r]["aggregated_loss"] for r in sorted(r2)]
            mono = all(l2[i] < l2[i - 1] for i in range(1, len(l2)))
            print(f"\nv2 monotonic decrease: {mono}; v2 reduction: {100 * (l2[0] - l2[-1]) / l2[0]:.1f}%; "
                  f"v2 total round time: {sum(r2[r]['elapsed_sec'] for r in r2):.1f}s")
            if r1:
                l1 = [r1[r]["aggregated_loss"] for r in sorted(r1)]
                print(f"v1 monotonic decrease: {all(l1[i] < l1[i-1] for i in range(1, len(l1)))}; "
                      f"v1 reduction: {100 * (l1[0] - l1[-1]) / l1[0]:.1f}%; v1 total round time: {sum(r1[r]['elapsed_sec'] for r in r1):.1f}s")

    # ---------------- non-IID simulation ----------------
    n1, n2 = load(v1d / "simulation_non_iid_results.json"), load(v2d / "simulation_non_iid_results.json")
    if n2:
        print("\n## run_simulation_non_iid: config (v2)\n")
        for k, v in n2["config"].items():
            print(f"- {k}: {v}")
        print("\n### Aggregated training loss per round (v1 uniform mean vs v2 sample-weighted)\n")
        keys = [("iid", "IID"), ("dirichlet_0.5", "Dir 0.5"), ("dirichlet_0.1", "Dir 0.1")]
        print("| Round | " + " | ".join(f"v1 {t} | v2 {t}" for _, t in keys) + " | v1 central | v2 central |")
        print("|---|" + "---|---|" * len(keys) + "---|---|")
        nr = len(n2["iid"]["rounds"])
        for i in range(nr):
            cells = []
            for k, _ in keys:
                cells.append(fmt(n1[k]["rounds"][i]["aggregated_loss"]) if n1 else "n/a")
                cells.append(fmt(n2[k]["rounds"][i]["aggregated_loss"]))
            c1 = fmt(n1["centralized"]["epochs"][i]["loss"]) if n1 else "n/a"
            c2 = fmt(n2["centralized"]["epochs"][i]["loss"])
            print(f"| {i + 1} | " + " | ".join(cells) + f" | {c1} | {c2} |")
        print("\n### v2 per-client losses and sample counts\n")
        for k, t in keys:
            recs = n2[k]["rounds"]
            cids = sorted(recs[0]["per_client_loss"])
            print(f"\n**{t}** (client n_k: {recs[0]['client_num_samples']})\n")
            print("| Round | agg (weighted) | agg (uniform) | " + " | ".join(cids) + " | elapsed (s) |")
            print("|---|---|---|" + "---|" * len(cids) + "---|")
            for r in recs:
                print(f"| {r['round']} | {fmt(r['aggregated_loss'])} | {fmt(r['aggregated_loss_uniform'])} | "
                      + " | ".join(fmt(r['per_client_loss'][c]) for c in cids) + f" | {r['elapsed_sec']} |")
            losses = [r["aggregated_loss"] for r in recs]
            print(f"monotonic: {all(losses[i] < losses[i-1] for i in range(1, len(losses)))}; "
                  f"reduction {100 * (losses[0] - losses[-1]) / losses[0]:.1f}%; wall {sum(r['elapsed_sec'] for r in recs):.1f}s")
        ce = n2["centralized"]["epochs"]
        print(f"\nCentralized: wall {sum(r['elapsed_sec'] for r in ce):.1f}s, steps/epoch {ce[0]['steps']}")

    # ---------------- accuracy ----------------
    a1, a2 = load(v1d / "non_iid_accuracy_results.json"), load(v2d / "non_iid_accuracy_results.json")
    if a2:
        meta = a2["_meta"]
        print("\n## non_iid_accuracy: meta (v2)\n")
        for k, v in meta.items():
            if k not in ("partitions", "metrics"):
                print(f"- {k}: {v}")
        for k, v in meta["partitions"].items():
            print(f"- partition {k}: sizes {[s['size'] for s in v]}")
        names = ["IID", "Dirichlet_0.5", "Dirichlet_0.1"]
        print("\n### Held-out test accuracy per round: v1 overall | v2 overall | v2 macro\n")
        print("| Round | " + " | ".join(f"{n} v1 | {n} v2 overall | {n} v2 macro" for n in names) + " |")
        print("|---|" + "---|---|---|" * len(names))
        for r in range(0, meta["num_rounds"] + 1):
            cells = []
            for n in names:
                v1r = next((x for x in a1[n] if x["round"] == r), None) if a1 else None
                v2r = next((x for x in a2[n] if x["round"] == r), None)
                cells.append(f"{v1r['test_accuracy'] * 100:.2f}%" if v1r else "n/a")
                cells.append(f"{v2r['accuracy_overall'] * 100:.2f}%" if v2r else "n/a")
                cells.append(f"{v2r['accuracy_macro'] * 100:.2f}%" if v2r else "n/a")
            print(f"| {r} | " + " | ".join(cells) + " |")
        print("\n### v2 aggregated (sample-weighted) training loss and elapsed per round\n")
        print("| Round | " + " | ".join(f"{n} loss | {n} elapsed (s)" for n in names) + " |")
        print("|---|" + "---|---|" * len(names))
        for r in range(1, meta["num_rounds"] + 1):
            cells = []
            for n in names:
                v2r = next(x for x in a2[n] if x["round"] == r)
                cells += [fmt(v2r["aggregated_loss"]), str(v2r["elapsed_sec"])]
            print(f"| {r} | " + " | ".join(cells) + " |")
        print(f"\nv2 total wall-clock: {meta['total_wall_sec']}s")
        print("\n### v2 per-class accuracy at round 5\n")
        print("| Class | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for c in range(10):
            print(f"| {c} | " + " | ".join(f"{a2[n][-1]['per_class_accuracy'][c] * 100:.2f}%" for n in names) + " |")


if __name__ == "__main__":
    main()

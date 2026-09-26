"""
Compare a v1 benchmark CSV with its v2 re-run (same cached generated clients,
fixed preflight validator).

Usage:
  python eval/compare_benchmark_versions.py \
      --v1 results/benchmark.csv --v2 results/benchmark_v2.csv \
      [--out results/benchmark_v1_v2_comparison.csv] [--md results/benchmark_v1_v2_comparison.md]

Prints / writes:
  1. per (script, method): v1 error_stage/e2e vs v2 error_stage/e2e, changed?
  2. strategy x stage-of-first-failure table (Syntax, S1 Import, S2 Hardware,
     S3 Data, S4 Forward, S5 Backward, E2E, Pass) for v1 and v2
  3. structured passes: data_path, data_path_exists, dataset_class/len
"""
import argparse
import csv
from collections import Counter, OrderedDict
from pathlib import Path

STAGE_COLS = OrderedDict([
    ("syntax", "Syntax"),
    ("interface", "Interface"),
    ("preflight/import_check", "S1 Import"),
    ("preflight/hardware_detect", "S2 Hardware"),
    ("preflight/data_load", "S3 Data"),
    ("preflight/forward_pass", "S4 Forward"),
    ("preflight/backward_pass", "S5 Backward"),
    ("e2e", "E2E"),
    ("", "Pass"),
])
METHODS = ["ast", "zero_shot", "few_shot", "structured"]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def key(r):
    return (r["script_name"], r["method"])


def first_line(msg):
    return (msg or "").strip().splitlines()[0] if msg else ""


def stage_table(rows, label):
    lines = [f"### Strategy x stage-of-first-failure ({label}, n={len(rows)} rows)", ""]
    header = "| Strategy | " + " | ".join(STAGE_COLS.values()) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(STAGE_COLS) + 1))
    for m in METHODS:
        mrows = [r for r in rows if r["method"] == m]
        c = Counter(r["error_stage"] for r in mrows)
        other = sum(v for k, v in c.items() if k not in STAGE_COLS)
        cells = [str(c.get(k, 0)) for k in STAGE_COLS]
        lines.append(f"| {m} | " + " | ".join(cells) + (f" | (other: {other})" if other else " |"))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1", required=True)
    p.add_argument("--v2", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--md", default=None)
    args = p.parse_args()

    v1 = {key(r): r for r in load(args.v1)}
    v2 = {key(r): r for r in load(args.v2)}
    keys = sorted(set(v1) | set(v2), key=lambda k: (METHODS.index(k[1]) if k[1] in METHODS else 9, k[0]))

    comp_rows = []
    for k in keys:
        a, b = v1.get(k), v2.get(k)
        row = {
            "script_name": k[0], "method": k[1],
            "framework": (b or a).get("framework", ""),
            "v1_error_stage": a["error_stage"] if a else "(missing)",
            "v1_e2e": a["e2e_runnable"] if a else "",
            "v1_error": first_line(a["error_message"]) if a else "",
            "v2_error_stage": b["error_stage"] if b else "(missing)",
            "v2_e2e": b["e2e_runnable"] if b else "",
            "v2_error": first_line(b["error_message"]) if b else "",
            "v2_dataset_class": b.get("dataset_class", "") if b else "",
            "v2_dataset_len": b.get("dataset_len", "") if b else "",
            "v2_data_path_exists": b.get("data_path_exists", "") if b else "",
            "v1_elapsed": a["elapsed_sec"] if a else "",
            "v2_elapsed": b["elapsed_sec"] if b else "",
        }
        row["stage_changed"] = row["v1_error_stage"] != row["v2_error_stage"]
        row["e2e_changed"] = row["v1_e2e"] != row["v2_e2e"]
        comp_rows.append(row)

    md = []
    md.append(f"# v1 vs v2 benchmark comparison\n\nv1: `{args.v1}`  \nv2: `{args.v2}`\n")
    md.append("## Per-row stage / e2e comparison\n")
    md.append("| script | method | v1 stage | v2 stage | v1 e2e | v2 e2e | stage changed | e2e changed | v2 error (first line) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in comp_rows:
        md.append(f"| {r['script_name']} | {r['method']} | {r['v1_error_stage'] or 'PASS'} | "
                  f"{r['v2_error_stage'] or 'PASS'} | {r['v1_e2e']} | {r['v2_e2e']} | "
                  f"{'YES' if r['stage_changed'] else ''} | {'YES' if r['e2e_changed'] else ''} | "
                  f"{r['v2_error'][:90].replace('|', '/')} |")
    n_stage = sum(r["stage_changed"] for r in comp_rows)
    n_e2e = sum(r["e2e_changed"] for r in comp_rows)
    md.append(f"\nRows with changed error_stage: {n_stage} / {len(comp_rows)}; "
              f"rows with changed e2e outcome: {n_e2e} / {len(comp_rows)}\n")
    if n_e2e:
        md.append("### Rows whose e2e outcome changed\n")
        for r in comp_rows:
            if r["e2e_changed"]:
                md.append(f"- {r['script_name']} / {r['method']}: v1 e2e={r['v1_e2e']} ({r['v1_error_stage']}: "
                          f"{r['v1_error'][:80]}) -> v2 e2e={r['v2_e2e']} ({r['v2_error_stage']}: {r['v2_error'][:80]})")
        md.append("")

    md.append(stage_table(list(v1.values()), "v1"))
    md.append("")
    md.append(stage_table(list(v2.values()), "v2"))
    md.append("")

    md.append("## Structured passes: data provenance (v2)\n")
    md.append("| script | framework | e2e | data_path | data_path_exists | dataset_class | dataset_len | elapsed (s) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(v2.values(), key=lambda r: (r["framework"], r["script_name"])):
        if r["method"] == "structured":
            md.append(f"| {r['script_name']} | {r['framework']} | {r['e2e_runnable']} | {r.get('data_path','')} | "
                      f"{r.get('data_path_exists','')} | {r.get('dataset_class','')} | {r.get('dataset_len','')} | {r['elapsed_sec']} |")
    text = "\n".join(md)
    print(text)

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
            w.writeheader(); w.writerows(comp_rows)
        print(f"\nSaved {args.out}")
    if args.md:
        Path(args.md).write_text(text + "\n")
        print(f"Saved {args.md}")


if __name__ == "__main__":
    main()

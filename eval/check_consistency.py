"""
Consistency check between archived results and the manuscript (reviewer-requested).

Regenerates the headline numbers from the archived CSV/JSON files and verifies
that each one appears in manuscript/main_tosem.tex (tables and text). Exits 1
on any mismatch. Run after every rerun and before every submission:

    python eval/check_consistency.py [--tex manuscript/main_tosem.tex] [--benchmark results/benchmark_v2.csv]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

STAGE_ORDER = ["preflight/import_check", "preflight/hardware_detect", "preflight/data_load",
               "preflight/forward_pass", "preflight/backward_pass"]
STRATEGY_TEX = {"ast": "AST", "template": "Template", "zero_shot": "Zero-shot", "few_shot": "Few-shot",
                "structured": "Structured"}


def stage_table(df: pd.DataFrame) -> dict[str, list[int]]:
    """strategy -> [S1, S2, S3, S4, S5, pass] over the primary scripts."""
    out = {}
    for m, g in df.groupby("method"):
        counts = [int((g.error_stage == s).sum()) for s in STAGE_ORDER]
        counts.append(int((g.e2e_runnable.astype(str) == "True").sum()))
        out[m] = counts
    return out


def tex_table_rows(tex: str, label: str) -> list[str]:
    """Return the tabular body lines of the table carrying \\label{label}."""
    i = tex.find(f"\\label{{{label}}}")
    if i < 0:
        return []
    start = tex.rfind("\\begin{table", 0, i)
    end = tex.find("\\end{table", i)
    return tex[start:end].splitlines()


def parse_stage_rows(lines: list[str]) -> dict[str, list[int]]:
    rows = {}
    for ln in lines:
        m = re.match(r"\s*(AST|Template|Zero-shot|Few-shot|Structured)\s*&(.*)\\\\", ln)
        if m:
            nums = [int(x) for x in re.findall(r"\d+", m.group(2))]
            rows[m.group(1)] = nums
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(ROOT / "manuscript" / "main_tosem.tex"))
    ap.add_argument("--benchmark", default=str(ROOT / "results" / "benchmark_v2.csv"))
    ap.add_argument("--sim", default=str(ROOT / "fl_results" / "v2" / "simulation_results.json"))
    ap.add_argument("--noniid", default=str(ROOT / "fl_results" / "v2" / "simulation_non_iid_results.json"))
    ap.add_argument("--acc", default=str(ROOT / "fl_results" / "v2" / "non_iid_accuracy_results.json"))
    a = ap.parse_args(argv)
    tex = Path(a.tex).read_text()
    problems: list[str] = []

    def must_contain(s: str, what: str):
        if s not in tex:
            problems.append(f"{what}: '{s}' not found in manuscript")

    # ── 1. primary benchmark: e2e counts and stage table ──────────────────
    df = pd.read_csv(a.benchmark)
    tpl = ROOT / "results" / "template_baseline.csv"
    if tpl.exists() and "template" not in set(df.method):
        t = pd.read_csv(tpl)
        t = t[t.script_name.isin(df.script_name.unique())]
        df = pd.concat([df, t[df.columns.intersection(t.columns)]], ignore_index=True)
    df = df[df.method.isin(STRATEGY_TEX)]
    n_scripts = df.script_name.nunique()
    st = stage_table(df)
    print(f"benchmark: {len(df)} rows, {n_scripts} scripts")
    for m, c in st.items():
        print(f"  {m:11s} S1..S5={c[:5]} pass={c[5]}")
        must_contain(f"{c[5]}/{n_scripts}", f"e2e count for {m}")
    rows = parse_stage_rows(tex_table_rows(tex, "tab:stages"))
    for m, c in st.items():
        name = STRATEGY_TEX[m]
        if name not in rows:
            problems.append(f"stage table row {name} missing in tab:stages")
            continue
        if rows[name] != c:
            problems.append(f"stage table row {name}: tex {rows[name]} != data {c}")
    # synthetic-data count among structured passes
    s = df[(df.method == "structured") & (df.e2e_runnable.astype(str) == "True")]
    if "dataset_class" in s:
        synth = s.dataset_class.astype(str).str.contains("Synthetic|TensorDataset|_Seq2Seq|FakeData", regex=True).sum()
        print(f"  structured passes on synthetic data: {synth}/{len(s)}")
    # McNemar for structured vs each baseline
    piv = df.pivot_table(index="script_name", columns="method", values="e2e_runnable",
                         aggfunc=lambda v: str(v.iloc[0]) == "True")
    from scipy.stats import binomtest
    for m in ("few_shot", "zero_shot", "ast", "template"):
        if m not in piv:
            continue
        b = int((piv["structured"] & ~piv[m]).sum()); c = int((~piv["structured"] & piv[m]).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")
        print(f"  McNemar structured vs {m}: discordant {b}+{c}, p={p:.4f}")
        if b + c:
            must_contain(f"{p:.4f}".rstrip("0"), f"McNemar p structured vs {m}")

    # ── 2. simulations ───────────────────────────────────────────────────
    sim = json.load(open(a.sim))
    for key, tag in (("pytorch_mnist", "Sim MNIST"), ("lightning_gan", "Sim GAN")):
        losses = [r["aggregated_loss"] for r in sim[key]["rounds"]]
        print(f"  {tag} losses: {[round(x, 4) for x in losses]}")
        for x in losses:
            must_contain(f"{x:.4f}", f"{tag} loss")
    non = json.load(open(a.noniid))
    for key, tag in (("iid", "IID"), ("dirichlet_0.5", "Dir0.5"), ("dirichlet_0.1", "Dir0.1")):
        losses = [r["aggregated_loss"] for r in non[key]["rounds"]]
        print(f"  non-IID {tag}: R1={losses[0]:.4f} R5={losses[-1]:.4f}")
        must_contain(f"{losses[0]:.4f}", f"non-IID {tag} R1 loss")
        must_contain(f"{losses[-1]:.4f}", f"non-IID {tag} R5 loss")
        sizes = [c["size"] for c in non[key]["partition_summary"]]
        if len(set(sizes)) == 1:          # equal shards are written as "20,000 examples each"
            must_contain(f"{sizes[0]:,}".replace(",", "{,}"), f"{tag} partition size")
        else:
            must_contain("/".join(f"{n:,}" for n in sizes).replace(",", "{,}"), f"{tag} partition sizes")
    acc = json.load(open(a.acc))
    for key in ("IID", "Dirichlet_0.5", "Dirichlet_0.1"):
        vals = [r["accuracy_overall"] * 100 for r in acc[key] if r["round"] > 0]
        print(f"  accuracy {key}: {[f'{v:.2f}' for v in vals]}")
        for v in vals:
            must_contain(f"{v:.2f}\\%", f"accuracy {key}")

    # ── 3. Phase-2 experiments (present only after those runs) ───────────
    from scipy.stats import beta as _beta

    def pct(k, n):
        return f"{100*k/n:.0f}"

    ph = ROOT / "results" / "self_repair_claude.csv"
    if ph.exists():
        d = pd.read_csv(ph); d["ok"] = d.e2e_runnable.astype(str) == "True"
        for base, g in d.groupby("base"):
            cells = g[["script_name", "seed"]].drop_duplicates().shape[0]
            final = int(g.groupby(["script_name", "seed"])["ok"].any().sum())
            print(f"  self-repair {base}: {final}/{cells} after the last round")
            must_contain(f"{final}/{cells}", f"self-repair {base} final")
    ph = ROOT / "results" / "ablation_claude.csv"
    if ph.exists():
        d = pd.read_csv(ph); d["ok"] = d.e2e_runnable.astype(str) == "True"
        for c, g in d.groupby("condition"):
            k, n = int(g.ok.sum()), len(g)
            print(f"  ablation {c}: {k}/{n}")
            must_contain(f"{k}/{n}", f"ablation {c}")
    ph = ROOT / "results" / "repeated_claude.csv"
    if ph.exists():
        d = pd.read_csv(ph)
        d = d[d.error_stage.astype(str) != "generation"]
        d["ok"] = d.e2e_runnable.astype(str) == "True"
        for m, g in d.groupby("method"):
            k, n = int(g.ok.sum()), len(g)
            print(f"  repeated claude {m}: {k}/{n}")
            must_contain(f"{k}/{n}", f"repeated claude {m}")
    ph = ROOT / "results" / "repeated_gemini.csv"
    if ph.exists():
        d = pd.read_csv(ph)
        d = d[d.error_stage.astype(str) != "generation"].copy()
        d["ok"] = d.e2e_runnable.astype(str) == "True"
        # only scripts with the complete five-sample grid under both strategies count
        per = d.groupby(["script_name", "method"])["sample"].nunique().unstack()
        full = per[(per >= 5).all(axis=1) & (per.notna().all(axis=1))].index
        g5 = d[d.script_name.isin(full)]
        print(f"  repeated gemini: {len(full)} scripts with the complete grid")
        for m, g in g5.groupby("method"):
            k, n = int(g.ok.sum()), len(g)
            print(f"  repeated gemini {m}: {k}/{n}")
            must_contain(f"{k}/{n}", f"repeated gemini {m}")
    ph = ROOT / "results" / "expansion.csv"
    tb = ROOT / "results" / "expansion_template_baseline.csv"
    if ph.exists() and tb.exists():
        d = pd.read_csv(ph); d["ok"] = d.e2e_runnable.astype(str) == "True"
        t = pd.read_csv(tb); t["ok"] = t.e2e_runnable.astype(str) == "True"
        piv = d[d.stratum == "holdout"].pivot_table(index="script_name", columns="method",
                                                    values="ok", aggfunc="first")
        piv["template"] = t[t.stratum == "holdout"].set_index("script_name")["ok"]
        n = len(piv)
        for m in piv.columns:
            k = int(piv[m].sum())
            print(f"  holdout {m}: {k}/{n}")
            must_contain(f"{k}/{n}", f"holdout {m}")
        for a2, b2 in [("structured", "template"), ("structured", "zero_shot")]:
            x = int((piv[a2] & ~piv[b2]).sum()); y = int((~piv[a2] & piv[b2]).sum())
            p = binomtest(x, x + y, 0.5).pvalue if x + y else float("nan")
            print(f"  McNemar holdout {a2} vs {b2}: {x}+{y}, p={p:.4f}")

    # ── report ───────────────────────────────────────────────────────────
    if problems:
        print("\nMISMATCHES:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nOK: all regenerated numbers appear in the manuscript.")


if __name__ == "__main__":
    main()

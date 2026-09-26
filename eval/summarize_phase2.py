"""
Summarize the Phase-2 experiment CSVs (self-repair, ablation, repeated sampling)
into paste-ready Markdown/LaTeX tables with Clopper-Pearson intervals and
per-generation token/cost figures.

    python eval/summarize_phase2.py [--self-repair results/self_repair.csv]
                                    [--ablation results/ablation.csv]
                                    [--repeated results/repeated.csv] [--out results/phase2_summary.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parent.parent


def cp_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def fmt_ci(k: int, n: int) -> str:
    lo, hi = cp_interval(k, n)
    return f"{k}/{n} ({100*k/n:.0f}%, CI [{100*lo:.1f}, {100*hi:.1f}])"


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower() == "true"


def summarize_self_repair(df: pd.DataFrame) -> list[str]:
    out = ["## Self-repair (harness feedback only, no schema)", ""]
    df = df.copy(); df["ok"] = truthy(df.e2e_runnable)
    for (prov, base), g in df.groupby(["provider", "base"]):
        n_cells = g[["script_name", "seed"]].drop_duplicates().shape[0]
        out.append(f"### {prov} / {base}  (cells = {n_cells})")
        out.append("| after round | cumulative passes | cumulative LLM calls (mean) | cumulative cost USD (mean) |")
        out.append("|---|---|---|---|")
        max_r = int(g["round"].max())
        for r in range(0, max_r + 1):
            # a cell counts as passed by round r if any row with round<=r passed
            passed = g[g["round"] <= r].groupby(["script_name", "seed"])["ok"].any().sum()
            last = g[g["round"] <= r].sort_values("round").groupby(["script_name", "seed"]).tail(1)
            out.append(f"| {r} | {fmt_ci(int(passed), n_cells)} | {last.cum_calls.astype(float).mean():.2f} | "
                       f"{last.cum_cost_usd.astype(float).mean():.4f} |")
        unresolved = g.sort_values("round").groupby(["script_name", "seed"]).tail(1)
        unresolved = unresolved[~unresolved.ok]
        if len(unresolved):
            out.append("")
            out.append("Unresolved after the last round: " + "; ".join(
                f"{r.script_name} ({r.error_stage})" for r in unresolved.itertuples()))
        out.append("")
    return out


def summarize_ablation(df: pd.DataFrame) -> list[str]:
    out = ["## Same-provider prompt ablation", ""]
    df = df.copy(); df["ok"] = truthy(df.e2e_runnable)
    for prov, g in df.groupby("provider"):
        out.append(f"### {prov}")
        out.append("| condition | system words | user words (mean) | generations | passes | per-script pass fraction mean +- sd | "
                   "mean in/out tokens | mean cost USD | McNemar vs schema_sys (script-level majority) |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        by_script = g.groupby(["condition", "script_name"])["ok"].mean().unstack(0)
        ref = (by_script["schema_sys"] >= 0.5) if "schema_sys" in by_script else None
        for cond, gc in g.groupby("condition"):
            k, n = int(gc.ok.sum()), len(gc)
            frac = by_script[cond]
            if ref is not None and cond != "schema_sys":
                cur = frac >= 0.5
                b = int((ref & ~cur).sum()); c = int((~ref & cur).sum())
                p = f"{b}+{c} disc., p={binomtest(b, b + c, 0.5).pvalue:.4f}" if b + c else "0 discordant"
            else:
                p = "ref"
            out.append(f"| {cond} | {gc.system_words.iloc[0]} | {gc.user_words.mean():.0f} | {n} | {fmt_ci(k, n)} | "
                       f"{frac.mean():.2f} +- {frac.std(ddof=0):.2f} | {gc.llm_input_tokens.mean():.0f}/{gc.llm_output_tokens.mean():.0f} | "
                       f"{gc.llm_cost_usd.astype(float).mean():.4f} | {p} |")
        out.append("")
    return out


def summarize_repeated(df: pd.DataFrame) -> list[str]:
    out = ["## Repeated sampling", ""]
    df = df.copy(); df["ok"] = truthy(df.e2e_runnable)
    for prov, g in df.groupby("provider"):
        out.append(f"### {prov}")
        out.append("| method | generations | passes | scripts always pass | scripts never pass | scripts mixed | "
                   "per-script pass fraction mean +- sd | mean in/out tokens | mean cost USD |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for m, gm in g.groupby("method"):
            frac = gm.groupby("script_name")["ok"].mean()
            out.append(f"| {m} | {len(gm)} | {fmt_ci(int(gm.ok.sum()), len(gm))} | {(frac == 1).sum()} | {(frac == 0).sum()} | "
                       f"{((frac > 0) & (frac < 1)).sum()} | {frac.mean():.2f} +- {frac.std(ddof=0):.2f} | "
                       f"{gm.llm_input_tokens.mean():.0f}/{gm.llm_output_tokens.mean():.0f} | {gm.llm_cost_usd.astype(float).mean():.4f} |")
        out.append("")
        piv = g.groupby(["script_name", "method"])["ok"].mean().unstack(1)
        out.append("Per-script pass fraction:")
        out.append(piv.round(2).to_markdown())
        out.append("")
    return out


def cost_table(frames: dict[str, pd.DataFrame]) -> list[str]:
    out = ["## Token and cost per generation (all Phase-2 rows with recorded usage)", ""]
    rows = []
    for name, df in frames.items():
        if df is None or "llm_input_tokens" not in df:
            continue
        d = df[(df.llm_cost_source.astype(str) != "dry_run") & df.llm_input_tokens.notna()]
        for (prov, m), g in d.groupby(["provider", "method"]):
            rows.append({"experiment": name, "provider": prov, "method": m, "n": len(g),
                         "in_tokens": g.llm_input_tokens.mean(), "out_tokens": g.llm_output_tokens.mean(),
                         "cost_usd": g.llm_cost_usd.astype(float).mean(), "sec": g.llm_elapsed_sec.astype(float).mean()})
    if rows:
        t = pd.DataFrame(rows).round({"in_tokens": 0, "out_tokens": 0, "cost_usd": 4, "sec": 1})
        out.append(t.to_markdown(index=False))
    else:
        out.append("(no live-call rows yet)")
    out.append("")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-repair", default=str(ROOT / "results" / "self_repair.csv"))
    ap.add_argument("--ablation", default=str(ROOT / "results" / "ablation.csv"))
    ap.add_argument("--repeated", default=str(ROOT / "results" / "repeated.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase2_summary.md"))
    a = ap.parse_args(argv)
    frames = {}
    lines = ["# Phase-2 summary", ""]
    for key, path, fn in (("self_repair", a.self_repair, summarize_self_repair),
                          ("ablation", a.ablation, summarize_ablation),
                          ("repeated", a.repeated, summarize_repeated)):
        p = Path(path)
        if p.exists():
            frames[key] = pd.read_csv(p)
            lines += fn(frames[key])
        else:
            frames[key] = None
            lines += [f"## {key}: {p} not found", ""]
    lines += cost_table(frames)
    Path(a.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

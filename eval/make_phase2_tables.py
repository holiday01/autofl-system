"""
Emit paste-ready LaTeX tables and a Markdown digest for the Phase-2 experiments:
self-repair, prompt ablation, repeated sampling, and the frozen expansion benchmark.

    python eval/make_phase2_tables.py [--out-tex results/phase2_tables.tex]
                                      [--out-md results/phase2_summary.md]

Every number is computed from the archived CSVs; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower() == "true"


def cp(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return 100 * lo, 100 * hi


def ci_tex(k: int, n: int) -> str:
    lo, hi = cp(k, n)
    return f"{k}/{n} & {100*k/n:.0f}\\% & [{lo:.1f}, {hi:.1f}]"


def mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    x = int((a & ~b).sum()); y = int((~a & b).sum())
    p = binomtest(x, x + y, 0.5).pvalue if x + y else float("nan")
    return x, y, p


def p_tex(p: float) -> str:
    if p != p:
        return "n/a"
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def load(name: str) -> pd.DataFrame | None:
    f = R / f"{name}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if "error_stage" in d:
        # rows recorded when the provider call itself failed are not observations
        d = d[d.error_stage.astype(str) != "generation"].copy()
    d["ok"] = truthy(d.e2e_runnable)
    return d


# ── tables ───────────────────────────────────────────────────────────────────

def t_self_repair(d: pd.DataFrame) -> tuple[str, list[str]]:
    rows, md = [], ["## Self-repair", ""]
    for base, g in d.groupby("base"):
        cells = g[["script_name", "seed"]].drop_duplicates().shape[0]
        line = [base.replace("_", "-")]
        for r in range(0, int(g["round"].max()) + 1):
            k = int(g[g["round"] <= r].groupby(["script_name", "seed"])["ok"].any().sum())
            line.append(f"{k}/{cells}")
        last = g.sort_values("round").groupby(["script_name", "seed"]).tail(1)
        line += [f"{last.cum_calls.astype(float).mean():.1f}",
                 f"{last.cum_cost_usd.astype(float).mean():.2f}"]
        rows.append(" & ".join(line) + r" \\")
        unresolved = last[~last.ok]
        md.append(f"- **{base}**: " + " -> ".join(line[1:-2]) +
                  f"; mean {line[-2]} repair calls, \\${line[-1]} per script; unresolved: " +
                  ("; ".join(f"{r.script_name} ({r.error_stage})" for r in unresolved.itertuples()) or "none"))
    tex = "\n".join(rows)
    return tex, md + [""]


def t_ablation(d: pd.DataFrame) -> tuple[str, list[str]]:
    by = d.groupby(["condition", "script_name"])["ok"].mean().unstack(0)
    ref = (by["schema_sys"] >= 0.5) if "schema_sys" in by else None
    order = ["schema_sys", "schema_user", "schema_sys_exemplar", "schema_sys_prohibit",
             "exemplar_invariants", "framework_rules_only"]
    label = {"schema_sys": "Schema in system prompt (reference)",
             "schema_user": "Schema in user message",
             "schema_sys_exemplar": "Schema plus corrected exemplar",
             "schema_sys_prohibit": "Schema, I3 phrased as a prohibition",
             "exemplar_invariants": "Exemplar plus invariants, no framework rules",
             "framework_rules_only": "Framework rules only, no invariants"}
    rows, md = [], ["## Prompt ablation", ""]
    for c in [c for c in order if c in set(d.condition)]:
        g = d[d.condition == c]
        k, n = int(g.ok.sum()), len(g)
        if ref is not None and c != "schema_sys":
            x, y, p = mcnemar(ref, by[c] >= 0.5)
            stat = f"{x}+{y} & {p_tex(p)}"
        else:
            stat = "-- & --"
        rows.append(f"{label[c]} & {ci_tex(k, n)} & {stat} \\\\")
        md.append(f"- **{label[c]}**: {k}/{n}")
    return "\n".join(rows), md + [""]


def t_repeated(frames: dict[str, pd.DataFrame], n_samples: int = 5) -> tuple[str, list[str]]:
    """Only scripts with the full sample grid for every method are counted, so a
    provider whose run was cut short by an API quota contributes a complete
    sub-grid rather than an uneven one."""
    rows, md = [], ["## Repeated sampling", ""]
    for prov, d in frames.items():
        cov = d.groupby(["script_name", "method"]).size().unstack(1).fillna(0)
        full = cov[(cov >= n_samples).all(axis=1)].index
        dropped = sorted(set(d.script_name) - set(full))
        d = d[d.script_name.isin(full)]
        md.append(f"- **{prov}**: {len(full)} scripts with the complete {n_samples}-sample grid"
                  + (f"; excluded for incomplete coverage: {', '.join(dropped)}" if dropped else ""))
        for m, g in d.groupby("method"):
            frac = g.groupby("script_name")["ok"].mean()
            k, n = int(g.ok.sum()), len(g)
            rows.append(f"{prov.capitalize()} & {m.replace('_', '-')} & {ci_tex(k, n)} & "
                        f"{int((frac == 1).sum())} & {int((frac == 0).sum())} & "
                        f"{int(((frac > 0) & (frac < 1)).sum())} \\\\")
            md.append(f"- **{prov} / {m}**: {k}/{n}; always {int((frac==1).sum())}, "
                      f"never {int((frac==0).sum())}, mixed {int(((frac>0)&(frac<1)).sum())}")
    return "\n".join(rows), md + [""]


def t_expansion(d: pd.DataFrame, t: pd.DataFrame) -> tuple[str, str, list[str]]:
    md = ["## Frozen expansion benchmark", ""]
    rows = []
    for stratum in ["dev_new", "holdout"]:
        piv = d[d.stratum == stratum].pivot_table(index="script_name", columns="method",
                                                  values="ok", aggfunc="first")
        piv["template"] = t[t.stratum == stratum].set_index("script_name")["ok"]
        n = len(piv)
        for m in ["template", "zero_shot", "few_shot_corrected", "structured"]:
            if m in piv:
                rows.append(f"{stratum.replace('_', '-')} & {m.replace('_', '-')} & {ci_tex(int(piv[m].sum()), n)} \\\\")
        md.append(f"- **{stratum}** (n={n}): " +
                  ", ".join(f"{m} {int(piv[m].sum())}/{n}" for m in
                            ["template", "zero_shot", "few_shot_corrected", "structured"] if m in piv))
    # contrasts on the holdout
    piv = d[d.stratum == "holdout"].pivot_table(index="script_name", columns="method",
                                                values="ok", aggfunc="first")
    piv["template"] = t[t.stratum == "holdout"].set_index("script_name")["ok"]
    crows = []
    for a, b in [("structured", "template"), ("structured", "few_shot_corrected"),
                 ("structured", "zero_shot"), ("template", "few_shot_corrected")]:
        x, y, p = mcnemar(piv[a], piv[b])
        crows.append(f"{a.replace('_', '-')} vs.\\ {b.replace('_', '-')} & {x} & {y} & {p_tex(p)} \\\\")
        md.append(f"- McNemar {a} vs {b}: {x}+{y} discordant, p={p_tex(p)}")
    adv = t[t.stratum == "adversarial"]
    md.append(f"- adversarial mutants (template): {int(adv.ok.sum())}/{len(adv)}")
    return "\n".join(rows), "\n".join(crows), md + [""]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tex", default=str(R / "phase2_tables.tex"))
    ap.add_argument("--out-md", default=str(R / "phase2_summary.md"))
    a = ap.parse_args(argv)

    tex, md = [], ["# Phase-2 results", ""]
    sr = load("self_repair_claude")
    if sr is not None:
        t, m = t_self_repair(sr); tex.append("%% self-repair rows\n" + t); md += m
    ab = load("ablation_claude")
    if ab is not None:
        t, m = t_ablation(ab); tex.append("%% ablation rows\n" + t); md += m
    reps = {p: load(f"repeated_{p}") for p in ("claude", "gemini")}
    reps = {k: v for k, v in reps.items() if v is not None}
    if reps:
        t, m = t_repeated(reps); tex.append("%% repeated-sampling rows\n" + t); md += m
    ex, tb = load("expansion"), load("expansion_template_baseline")
    if ex is not None and tb is not None:
        t1, t2, m = t_expansion(ex, tb)
        tex.append("%% expansion rows\n" + t1); tex.append("%% expansion contrasts\n" + t2); md += m
    # cost
    costs = []
    for name, d in [("self-repair", sr), ("ablation", ab), ("repeated-claude", reps.get("claude")),
                    ("repeated-gemini", reps.get("gemini")), ("expansion", ex)]:
        if d is None or "llm_cost_usd" not in d:
            continue
        g = d[(d.llm_cost_source.astype(str).isin(["provider", "price_table"]))]
        for (prov,), gg in g.groupby(["llm_provider"] if "llm_provider" in g else ["provider"]):
            costs.append(f"{name} & {prov} & {len(gg)} & {gg.llm_input_tokens.mean():.0f} & "
                         f"{gg.llm_output_tokens.mean():.0f} & {gg.llm_cost_usd.astype(float).mean():.3f} \\\\")
    if costs:
        tex.append("%% cost rows\n" + "\n".join(costs))
        md += ["## Cost per generation", ""] + ["- " + c.replace(" \\\\", "").replace(" & ", " | ") for c in costs]

    Path(a.out_tex).write_text("\n\n".join(tex) + "\n")
    Path(a.out_md).write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[wrote {a.out_tex} and {a.out_md}]")


if __name__ == "__main__":
    main()

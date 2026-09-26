"""Generate publication-quality figures from a benchmark CSV.

Usage:
  python eval/plot_results.py [--input results/benchmark.csv] [--out-dir results/figures]

The method list includes the rule-based template converter ("template");
methods absent from the input CSV are drawn with a 0 rate, so pass a CSV
that contains every listed method (e.g. results/benchmark_v2_with_template.csv).
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"          # overridden by --out-dir

METHODS = ["ast", "template", "zero_shot", "few_shot", "structured"]
METHOD_LABELS = ["AST", "Template", "Zero-shot", "Few-shot", "Structured"]
FRAMEWORKS = ["pytorch", "tensorflow", "monai", "lightning"]
FW_LABELS = ["PyTorch", "TensorFlow", "MONAI", "Lightning"]
COLORS = ["#4C72B0", "#8172B2", "#DD8452", "#55A868", "#C44E52"]


def load(path: str | Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def _orig(rows):
    """Return only rows for original scripts (exclude pre-generated _fl_ files)."""
    return [x for x in rows if "_fl_" not in x["script_name"]]


def rate(rows, fw, method, col):
    r = [x for x in _orig(rows) if x["framework"] == fw and x["method"] == method]
    if not r:
        return 0.0
    return sum(x[col] == "True" for x in r) / len(r)


def mean_cov(rows, fw, method):
    r = [x for x in _orig(rows) if x["framework"] == fw and x["method"] == method]
    if not r:
        return 0.0
    return sum(float(x["component_coverage"]) for x in r) / len(r)


def fig1_e2e_heatmap(rows):
    """Heatmap: e2e_runnable rate — frameworks × methods."""
    data = np.array([
        [rate(rows, fw, m, "e2e_runnable") for m in METHODS]
        for fw in FRAMEWORKS
    ])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHOD_LABELS, fontsize=11)
    ax.set_yticks(range(len(FRAMEWORKS))); ax.set_yticklabels(FW_LABELS, fontsize=11)
    for i in range(len(FRAMEWORKS)):
        for j in range(len(METHODS)):
            ax.text(j, i, f"{data[i,j]:.0%}", ha="center", va="center",
                    fontsize=12, color="black" if data[i,j] < 0.7 else "white",
                    fontweight="bold")
    plt.colorbar(im, ax=ax, label="e2e success rate")
    ax.set_title("End-to-end FL Runnable Rate by Skill × Framework", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_e2e_heatmap.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_e2e_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig1_e2e_heatmap")


def fig2_coverage_grouped(rows):
    """Grouped bar: component_coverage per method × framework."""
    x = np.arange(len(FRAMEWORKS))
    width = 0.8 / len(METHODS)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (m, label, color) in enumerate(zip(METHODS, METHOD_LABELS, COLORS)):
        vals = [mean_cov(rows, fw, m) for fw in FRAMEWORKS]
        ax.bar(x + (i - (len(METHODS) - 1) / 2) * width, vals, width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(FW_LABELS, fontsize=11)
    ax.set_ylabel("Component Coverage (mean)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title("Component Coverage by Skill × Framework", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_coverage_grouped.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_coverage_grouped.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig2_coverage_grouped")


def fig3_stage_breakdown(rows):
    """Grouped bar: pass rate at each pipeline stage, per method × framework.

    Zero-rate bars are rendered as a small visible nub (~2% height) so the
    structural-vs-behavioral asymmetry is readable; readers should treat
    any bar shorter than the dashed 10% reference line as 0% pass.
    """
    stage_cols = [
        ("interface_complete", "Interface Complete", "#4393c3"),
        ("preflight_pass",     "Preflight Pass",     "#92c5de"),
        ("e2e_runnable",       "E2E Runnable",       "#1a9641"),
    ]
    fig, axes = plt.subplots(1, len(FRAMEWORKS), figsize=(13, 4.2), sharey=True)
    width = 0.22
    x = np.arange(len(METHODS))
    nub = 0.022  # baseline nub so 0% bars stay visible
    for ax, fw, fw_label in zip(axes, FRAMEWORKS, FW_LABELS):
        for i, (col, label, color) in enumerate(stage_cols):
            raw = np.array([rate(rows, fw, m, col) for m in METHODS])
            visible = np.where(raw > 0, raw, nub)
            ax.bar(x + (i - 1) * width, visible, width, label=label,
                   color=color, alpha=0.88,
                   edgecolor="black", linewidth=0.4)
        ax.set_title(fw_label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_LABELS, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(-0.03, 1.18)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
    axes[0].set_ylabel("Pass Rate", fontsize=11)
    handles = [mpatches.Patch(color=c, label=l) for _, l, c in stage_cols]
    fig.legend(handles=handles, loc="lower center", fontsize=10,
               bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    fig.suptitle("Pipeline Stage Pass Rates by Strategy $\\times$ Framework",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(FIG_DIR / "fig3_stage_breakdown.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_stage_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig3_stage_breakdown")


def fig4_per_script(rows):
    """Per-script success heatmap across 10 original scripts × len(METHODS) methods."""
    scripts = sorted(set(r["script_name"] for r in _orig(rows)))
    data = np.array([
        [1 if any(
            x["e2e_runnable"] == "True"
            for x in rows if x["script_name"] == s and x["method"] == m
        ) else 0
         for m in METHODS]
        for s in scripts
    ])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHOD_LABELS, fontsize=10)
    ax.set_yticks(range(len(scripts)))
    ax.set_yticklabels([s.replace("_main", "").replace("_tutorial", "")
                        .replace("_from_scratch", "_scratch")
                        .replace("_segmentation_3d", "_3d") for s in scripts], fontsize=9)
    for i in range(len(scripts)):
        for j in range(len(METHODS)):
            ax.text(j, i, "✓" if data[i, j] else "✗", ha="center", va="center",
                    fontsize=12, color="white" if data[i, j] else "black")
    ax.set_title("Per-Script E2E Success", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_per_script.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_per_script.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved fig4_per_script")


def main(argv=None):
    global FIG_DIR
    p = argparse.ArgumentParser(description="AutoFL benchmark figures")
    p.add_argument("--input", default=str(RESULTS_DIR / "benchmark.csv"))
    p.add_argument("--out-dir", default=str(FIG_DIR))
    args = p.parse_args(argv)
    FIG_DIR = Path(args.out_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"benchmark CSV not found at {csv_path}")
        sys.exit(1)
    rows = load(csv_path)
    print(f"Loaded {len(rows)} rows")

    fig1_e2e_heatmap(rows)
    fig2_coverage_grouped(rows)
    fig3_stage_breakdown(rows)
    fig4_per_script(rows)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()

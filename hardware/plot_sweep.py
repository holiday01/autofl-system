"""
Generate publication-quality heatmap figure from hardware_sweep.csv.
Outputs: results/figures/fig6_hardware_sweep.pdf  and  .png
"""

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH  = PROJECT_ROOT / "results/hardware_sweep.csv"
FIG_DIR   = PROJECT_ROOT / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── detector recommendation ──────────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
from hardware.detector import detect
profile  = detect()
DET_BS   = profile.suggested_batch_size   # 16
DET_LR   = 3e-4                           # CPU default (detector has no lr field)

# ── load CSV ─────────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "batch_size":       int(row["batch_size"]),
            "lr":               float(row["lr"]),
            "avg_loss_round2":  float(row["avg_loss_round2"]),
            "steps_per_sec":    float(row["steps_per_sec"]),
        })

batch_sizes    = sorted(set(r["batch_size"] for r in rows))
lrs            = sorted(set(r["lr"]         for r in rows))
n_bs, n_lr     = len(batch_sizes), len(lrs)

loss_mat = np.full((n_bs, n_lr), np.nan)
sps_mat  = np.full((n_bs, n_lr), np.nan)

for r in rows:
    i = batch_sizes.index(r["batch_size"])
    j = lrs.index(r["lr"])
    loss_mat[i, j] = r["avg_loss_round2"]
    sps_mat[i, j]  = r["steps_per_sec"]

# ── figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
fig.suptitle("Hardware Parameter Sweep — MNIST FL (CPU, 2 rounds, 1 epoch)",
             fontsize=10, fontweight="bold", y=1.01)

lr_labels  = [f"{lr:.0e}" for lr in lrs]
bs_labels  = [str(bs) for bs in batch_sizes]

def draw_heatmap(ax, mat, title, cmap, fmt=".3f", annot_color_thresh=None):
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    im = ax.imshow(mat, cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)

    ax.set_xticks(range(n_lr))
    ax.set_yticks(range(n_bs))
    ax.set_xticklabels(lr_labels, rotation=30, ha="right")
    ax.set_yticklabels(bs_labels)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Batch Size")
    ax.set_title(title)

    # Annotate each cell
    mid = (vmin + vmax) / 2
    for i in range(n_bs):
        for j in range(n_lr):
            val = mat[i, j]
            if np.isnan(val):
                continue
            txt_color = "white" if (val < mid) else "black"
            if annot_color_thresh is not None:
                txt_color = "white" if (val > annot_color_thresh) else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=7, color=txt_color, fontweight="normal")

    return im

# --- panel 1: avg_loss ---
draw_heatmap(axes[0], loss_mat,
             title="Final-Round Avg Loss",
             cmap="RdYlGn_r",
             fmt=".4f")

# --- panel 2: steps/sec ---
draw_heatmap(axes[1], sps_mat,
             title="Steps per Second",
             cmap="YlGnBu",
             fmt=".1f",
             annot_color_thresh=np.nanpercentile(sps_mat, 60))

# ── mark detector recommendation ─────────────────────────────────────────────
for ax, mat in zip(axes, [loss_mat, sps_mat]):
    if DET_BS in batch_sizes and DET_LR in lrs:
        row_i = batch_sizes.index(DET_BS)
        col_j = lrs.index(DET_LR)
        ax.plot(col_j, row_i, marker="*", markersize=14,
                color="gold", markeredgecolor="black", markeredgewidth=0.8,
                zorder=5, label=f"Detector rec. (bs={DET_BS}, lr={DET_LR:.0e})")
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85)

plt.tight_layout()

for ext in ("pdf", "png"):
    out_path = FIG_DIR / f"fig6_hardware_sweep.{ext}"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close()
print("Done.")

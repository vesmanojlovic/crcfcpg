#!/usr/bin/env python3
"""Scatter matrix and distance heatmap figure per tumour, for data and simulations."""
# usage: python scripts/final_tumour_figure.py [D E ...]  (default: all tumours)

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data/individual"
SIM_DIR = REPO_DIR / "figures/tumour_pairs"
FIG_DIR = REPO_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALL_TUMOURS = ["D", "E", "F", "I", "J", "M", "P", "S", "U", "X"]
SIDE_SPLIT = {"D": (2, 6), "P": (3, 5)}
HEATMAP_VMAX = 0.15

YELLOW_TO_PURPLE = LinearSegmentedColormap.from_list(
    "yellow_purple", ["#fbbf24", "#f59e0b", "#7c3aed", "#4c1d95", "#1e1b4b"]
)


def load_real_data(tid):
    df = pd.read_csv(DATA_DIR / f"tumour_{tid}.csv")
    gland_cols = [c for c in df.columns if c != "Labels"]
    side_a = sorted([c for c in gland_cols if len(c) >= 2 and c[1].upper() == "A"])
    side_b = sorted([c for c in gland_cols if len(c) >= 2 and c[1].upper() == "B"])
    ordered = side_a + side_b
    n_a = len(side_a)
    sub = df[ordered].dropna()
    arrays = {c: sub[c].values for c in ordered}
    return ordered, arrays, n_a


def load_sim_data(tid):
    npz = np.load(SIM_DIR / f"sim_data_{tid}.npz", allow_pickle=True)
    gland_names = list(npz["gland_names"])
    arrays = {n: npz[n] for n in gland_names}
    n_a = SIDE_SPLIT.get(tid, (4, 4))[0]
    return gland_names, arrays, n_a


def make_figure(gland_names, arrays, tid, fig_type, n_a):
    n = len(gland_names)

    fig = plt.figure(figsize=(14, 8))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.55], wspace=0.25)

    inner = outer[0].subgridspec(n, n, hspace=0.08, wspace=0.08)
    bins_scatter = np.linspace(0, 1, 31)

    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            ax = fig.add_subplot(inner[i, j])
            if i == j:
                ax.hist(arrays[gland_names[i]], bins=bins_scatter,
                        color="#2563eb", edgecolor="white", linewidth=0.2)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
            else:
                x = arrays[gland_names[j]]
                y = arrays[gland_names[i]]
                abs_diff = np.abs(x - y)
                ax.scatter(x, y, s=3, c=abs_diff, cmap='viridis_r',
                           vmin=0, vmax=0.25, lw=0, rasterized=True)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(gland_names[i], fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=6)
            if i == n - 1:
                ax.set_xlabel(gland_names[j], fontsize=10, rotation=45,
                              ha="right", va="top", labelpad=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#64748b")

    # single gland histogram in the empty upper triangle
    inset_r = min(3, n // 2)
    inset_c = min(3, n // 2)
    ax_hist = fig.add_subplot(inner[0:inset_r, n - inset_c:n])
    ax_hist.hist(arrays[gland_names[0]], bins=40, range=(0, 1),
                 color="#2563eb", edgecolor="white", linewidth=0.4)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel(r"fCpG $\beta$", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title(f"Single gland ({gland_names[0]})", fontsize=11)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.tick_params(labelsize=9)

    # overlay to draw the side A/B separator across the whole grid
    ax_ov = fig.add_subplot(outer[0], frameon=False)
    ax_ov.set_xticks([])
    ax_ov.set_yticks([])
    ax_ov.set_xlim(0, n)
    ax_ov.set_ylim(n, 0)
    ax_ov.axhline(n_a, color="#c53030", lw=1.2, xmin=0, xmax=n_a / n)
    ax_ov.axvline(n_a, color="#c53030", lw=1.2,
                  ymin=1 - n / n, ymax=1 - n_a / n)
    ax_ov.patch.set_alpha(0)

    ax_c = fig.add_subplot(outer[1])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.mean(
                (arrays[gland_names[i]] - arrays[gland_names[j]]) ** 2)
    im = ax_c.imshow(D, cmap="magma", vmin=0, vmax=HEATMAP_VMAX)
    ax_c.set_xticks(range(n))
    ax_c.set_yticks(range(n))
    ax_c.set_xticklabels(gland_names, rotation=45, ha="right", fontsize=10)
    ax_c.set_yticklabels(gland_names, fontsize=10)
    ax_c.axhline(n_a - 0.5, color="#ffffff", lw=1.2)
    ax_c.axvline(n_a - 0.5, color="#ffffff", lw=1.2)
    plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04,
                 label=r"Mean-squared $\beta$ difference")

    out_pdf = FIG_DIR / f"fig_{fig_type}_{tid}_final.pdf"
    out_png = FIG_DIR / f"fig_{fig_type}_{tid}_final.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    return out_pdf


def process_tumour(tid):
    print(f"Loading data for tumour {tid}...")
    gland_names, arrays, n_a = load_real_data(tid)
    print(f"{len(gland_names)} glands, {n_a}A + "
          f"{len(gland_names) - n_a}B")
    make_figure(gland_names, arrays, tid, "data", n_a)

    sim_path = SIM_DIR / f"sim_data_{tid}.npz"
    if sim_path.exists():
        print(f"Loading sim data for tumour {tid}...")
        sim_names, sim_arrays, n_a_sim = load_sim_data(tid)
        print(f"{len(sim_names)} glands")
        make_figure(sim_names, sim_arrays, tid, "sim", n_a_sim)
    else:
        print(f"No sim data for tumour {tid}, skipping")


def main():
    import sys
    tids = sys.argv[1:] if len(sys.argv) > 1 else ALL_TUMOURS
    for tid in tids:
        process_tumour(tid)


if __name__ == "__main__":
    main()

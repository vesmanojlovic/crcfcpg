#!/usr/bin/env python3
"""Data figure for tumour U."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

TUMOUR = "U"
REPO_DIR = Path(__file__).resolve().parents[1]
DATA = (REPO_DIR / "data/individual"
        / f"tumour_{TUMOUR}.csv")
FIG_DIR = REPO_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

YELLOW_TO_PURPLE = LinearSegmentedColormap.from_list(
    "yellow_purple", ["#fbbf24", "#f59e0b", "#7c3aed", "#4c1d95", "#1e1b4b"]
)


def load_tumour():
    df = pd.read_csv(DATA)
    gland_cols = [c for c in df.columns if c != "Labels"]
    # 2nd char indicates side A/B
    side_a = sorted([c for c in gland_cols
                     if len(c) >= 2 and c[1].upper() == "A"])
    side_b = sorted([c for c in gland_cols
                     if len(c) >= 2 and c[1].upper() == "B"])
    ordered = side_a + side_b
    sub = df[ordered].dropna()
    arrays = {c: sub[c].values for c in ordered}
    return ordered, arrays


def panel_A(ax, arrays, gland):
    beta = arrays[gland]
    ax.hist(beta, bins=40, range=(0, 1), color="#2563eb",
            edgecolor="white", linewidth=0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"fCpG $\beta$")
    ax.set_ylabel("Count")
    ax.set_title(f"A. Single gland ({gland})", fontsize=11, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_B(fig, outer_gs, gland_names, arrays):
    n = len(gland_names)
    inner = outer_gs.subgridspec(n, n, hspace=0.08, wspace=0.08)

    bins = np.linspace(0, 1, 31)

    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            ax = fig.add_subplot(inner[i, j])
            if i == j:
                ax.hist(arrays[gland_names[i]], bins=bins,
                        color="#2563eb", edgecolor="white", linewidth=0.2)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
            else:
                x = arrays[gland_names[j]]
                y = arrays[gland_names[i]]
                dist = np.abs(x - y) / np.sqrt(2)
                ax.scatter(x, y, c=dist, cmap=YELLOW_TO_PURPLE,
                           s=0.3, alpha=0.6, edgecolors="none",
                           vmin=0, vmax=0.7, rasterized=True)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_facecolor("white")

            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(gland_names[i], fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=4)
            if i == n - 1:
                ax.set_xlabel(gland_names[j], fontsize=7, rotation=45,
                              ha="right", va="top", labelpad=2)

            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#64748b")

    # transparent overlay axes to draw the side A/B dividers across the block
    ax_overlay = fig.add_subplot(outer_gs, frameon=False)
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])
    ax_overlay.set_xlim(0, n)
    ax_overlay.set_ylim(n, 0)
    ax_overlay.set_title(f"B. Paired gland $\\beta$ densities (tumour {TUMOUR})",
                         fontsize=11, loc="left", pad=12)
    ax_overlay.axhline(4, color="#c53030", lw=1.2, xmin=0, xmax=4 / n)
    ax_overlay.axvline(4, color="#c53030", lw=1.2, ymin=1 - 8 / n,
                       ymax=1 - 4 / n)
    ax_overlay.patch.set_alpha(0)


def panel_C(ax, gland_names, arrays):
    n = len(gland_names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.mean(
                (arrays[gland_names[i]] - arrays[gland_names[j]]) ** 2
            )
    im = ax.imshow(D, cmap="magma")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(gland_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(gland_names, fontsize=7)
    # side A/B divider
    ax.axhline(3.5, color="#ffffff", lw=1.2)
    ax.axvline(3.5, color="#ffffff", lw=1.2)
    ax.set_title(f"C. Inter-gland distance (tumour {TUMOUR})", fontsize=11, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean sq. diff.")


def main():
    gland_names, arrays = load_tumour()
    print(f"Loaded tumour {TUMOUR}: {len(gland_names)} glands, "
          f"{len(next(iter(arrays.values())))} fCpG loci")
    print(f"Order: {gland_names}")

    fig = plt.figure(figsize=(14, 6.5))
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.0, 1.6, 1.2],
        height_ratios=[1, 1],
        wspace=0.35, hspace=0.45,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    panel_A(ax_a, arrays, gland_names[0])

    panel_B(fig, gs[:, 1], gland_names, arrays)

    ax_c = fig.add_subplot(gs[:, 2])
    panel_C(ax_c, gland_names, arrays)

    # empty slot under panel A
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")

    out_pdf = FIG_DIR / "fig_data.pdf"
    out_png = FIG_DIR / "fig_data.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

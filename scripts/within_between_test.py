#!/usr/bin/env python3
"""Wilcoxon test across tumours of whether within-side gland distances are lower than between-side."""

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data/individual"
OUTPUT_DIR = REPO_DIR / "outputs"
FIGURE_DIR = REPO_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TUMOUR_IDS = ["D", "E", "F", "I", "J", "M", "P", "S", "U", "X"]

ADENOMAS = {'P', 'S', 'X'}
CARCINOMA_COLOR = 'steelblue'
ADENOMA_COLOR = 'coral'


def assign_side(col_name: str, tumour_id: str) -> str:
    # char after the tumour ID gives the side, e.g. 'EA LL 1' -> A, 'Hb3' -> B
    name_upper = col_name.upper().strip()
    if not name_upper.startswith(tumour_id.upper()):
        raise ValueError(
            f"Column '{col_name}' does not start with tumour ID '{tumour_id}'"
        )
    second_char = name_upper[len(tumour_id)]
    if second_char == "A":
        return "A"
    elif second_char == "B":
        return "B"
    else:
        raise ValueError(
            f"Cannot determine side for column '{col_name}' "
            f"(second char after tumour ID is '{second_char}')"
        )


def compute_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Mean squared beta difference between each pair of glands (columns of df)."""
    glands = df.columns.tolist()
    n = len(glands)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            msd = np.nanmean((df.iloc[:, i].values - df.iloc[:, j].values) ** 2)
            dist[i, j] = msd
            dist[j, i] = msd
    return pd.DataFrame(dist, index=glands, columns=glands)


def split_distances(dist_df: pd.DataFrame, sides: dict) -> tuple:
    """
    Split pairwise distances into within-side and between-side.
    Args:
        dist_df: symmetric distance matrix
        sides: dict of gland name -> 'A' or 'B'
    Returns:
        within, between: lists of distances
    """
    glands = dist_df.columns.tolist()
    within = []
    between = []
    for i, j in combinations(range(len(glands)), 2):
        d = dist_df.iloc[i, j]
        if sides[glands[i]] == sides[glands[j]]:
            within.append(d)
        else:
            between.append(d)
    return within, between


def main():
    results = {}
    tumour_within_medians = []
    tumour_between_medians = []
    tumour_labels = []

    for tid in TUMOUR_IDS:
        fpath = DATA_DIR / f"tumour_{tid}.csv"
        if not fpath.exists():
            print(f"{fpath} not found, skipping tumour {tid}")
            continue

        df = pd.read_csv(fpath, index_col=0)
        glands = df.columns.tolist()

        if len(glands) != 8:
            print(f"Tumour {tid} has {len(glands)} glands, expected 8")

        sides = {}
        for g in glands:
            sides[g] = assign_side(g, tid)

        n_a = sum(1 for v in sides.values() if v == "A")
        n_b = sum(1 for v in sides.values() if v == "B")

        dist_df = compute_distance_matrix(df)

        within, between = split_distances(dist_df, sides)

        med_within = float(np.median(within))
        med_between = float(np.median(between))
        mean_within = float(np.mean(within))
        mean_between = float(np.mean(between))

        pattern_holds = med_within < med_between
        ratio = med_between / med_within if med_within > 0 else float("inf")

        results[tid] = {
            "n_side_A": n_a,
            "n_side_B": n_b,
            "n_within_pairs": len(within),
            "n_between_pairs": len(between),
            "median_within": med_within,
            "median_between": med_between,
            "mean_within": mean_within,
            "mean_between": mean_between,
            "between_to_within_ratio": ratio,
            "within_lt_between": pattern_holds,
        }

        tumour_within_medians.append(med_within)
        tumour_between_medians.append(med_between)
        tumour_labels.append(tid)

        print(
            f"Tumour {tid}: median within={med_within:.6f}, "
            f"median between={med_between:.6f}, "
            f"ratio={ratio:.3f}, pattern={'YES' if pattern_holds else 'NO'}"
        )

    within_arr = np.array(tumour_within_medians)
    between_arr = np.array(tumour_between_medians)
    differences = between_arr - within_arr

    n_tumours = len(tumour_labels)
    n_pattern_holds = sum(1 for r in results.values() if r["within_lt_between"])

    stat, p_value = wilcoxon(
        within_arr, between_arr, alternative="less"
    )

    # rank-biserial r = 1 - 2T+ / (n(n+1)/2); scipy returns T+ for alternative='less'
    n = len(differences)
    T_plus = stat
    max_T = n * (n + 1) / 2
    rank_biserial_r = 1.0 - (2.0 * T_plus / max_T)

    median_diff = float(np.median(differences))
    mean_diff = float(np.mean(differences))

    overall = {
        "n_tumours": n_tumours,
        "n_pattern_holds": n_pattern_holds,
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p_value": float(p_value),
        "rank_biserial_r": float(rank_biserial_r),
        "median_difference_between_minus_within": median_diff,
        "mean_difference_between_minus_within": mean_diff,
        "tumour_order": tumour_labels,
        "median_within_values": [float(x) for x in tumour_within_medians],
        "median_between_values": [float(x) for x in tumour_between_medians],
    }

    print(f"Wilcoxon signed-rank test (one-sided: within < between)")
    print(f"  n = {n_tumours} tumours")
    print(f"  Pattern holds (within < between): {n_pattern_holds}/{n_tumours}")
    print(f"  Test statistic T+ = {stat:.1f}")
    print(f"  p-value = {p_value:.6f}")
    print(f"  Rank-biserial r = {rank_biserial_r:.4f}")
    print(f"  Median difference (between - within) = {median_diff:.6f}")

    outliers = [tid for tid in tumour_labels if not results[tid]["within_lt_between"]]
    if outliers:
        print(f"Tumours with within >= between: {', '.join(outliers)}")
    else:
        print("Within < between for all tumours")

    output = {"per_tumour": results, "overall": overall}
    out_path = OUTPUT_DIR / "within_between_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}")

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, tid in enumerate(tumour_labels):
        w = tumour_within_medians[i]
        b = tumour_between_medians[i]
        color = ADENOMA_COLOR if tid in ADENOMAS else CARCINOMA_COLOR
        ax.plot(
            [0, 1], [w, b],
            marker="o", color=color, linewidth=1.5,
            markersize=7, alpha=0.7, zorder=2,
        )

    # spread out the tumour labels so they do not overlap
    label_data = sorted(
        [(tumour_between_medians[i], tumour_labels[i])
         for i in range(len(tumour_labels))],
        key=lambda x: x[0],
    )
    y_range = (max(tumour_between_medians + tumour_within_medians)
               - min(tumour_between_medians + tumour_within_medians))
    min_gap = y_range * 0.045
    adjusted_y = [ld[0] for ld in label_data]
    for k in range(1, len(adjusted_y)):
        if adjusted_y[k] - adjusted_y[k - 1] < min_gap:
            adjusted_y[k] = adjusted_y[k - 1] + min_gap
    shift = np.mean([ld[0] for ld in label_data]) - np.mean(adjusted_y)
    adjusted_y = [y + shift for y in adjusted_y]

    for (orig_y, tid), adj_y in zip(label_data, adjusted_y):
        kw = dict(fontsize=10, color="black", fontweight="bold", va="center")
        if abs(adj_y - orig_y) > min_gap * 0.2:
            kw["arrowprops"] = dict(arrowstyle="-", color="grey", lw=0.5)
        ax.annotate(tid, xy=(1, orig_y), xytext=(1.06, adj_y), **kw)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within-side\n(median MSD)", "Between-side\n(median MSD)"],
                       fontsize=12)
    ax.set_ylabel("Median pairwise distance (MSD)", fontsize=12)
    ax.set_title(
        f"Within-side vs between-side inter-gland distances\n"
        f"Wilcoxon signed-rank p = {p_value:.4f}, r = {rank_biserial_r:.2f}",
        fontsize=13,
    )
    ax.set_xlim(-0.3, 1.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=CARCINOMA_COLOR, lw=1.5,
               markersize=7, label='Carcinoma'),
        Line2D([0], [0], marker='o', color=ADENOMA_COLOR, lw=1.5,
               markersize=7, label='Adenoma'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left',
              frameon=True, framealpha=0.9)

    plt.tight_layout()

    fig_path_png = FIGURE_DIR / "fig_within_between_test.png"
    fig_path_pdf = FIGURE_DIR / "fig_within_between_test.pdf"
    fig.savefig(fig_path_png, dpi=300)
    fig.savefig(fig_path_pdf)
    plt.close(fig)
    print(f"Saved {fig_path_png} and {fig_path_pdf}")


if __name__ == "__main__":
    main()

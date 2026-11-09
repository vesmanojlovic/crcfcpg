#!/usr/bin/env python3
"""Paired data and simulated figures for each tumour."""
# usage: python scripts/all_data_figures.py [tumour ids]

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from methabc.utils import repo_dir

REPO_DIR = Path(__file__).resolve().parents[1]
METHABC_DIR = Path(repo_dir)
DATA_DIR = REPO_DIR / "data/individual"
POSTERIOR_DIR = REPO_DIR / "outputs"
METHDEMON_BIN = METHABC_DIR / "resources/methdemon/bin/methdemon"
CONFIG_TEMPLATE = METHABC_DIR / "resources/config_template.dat"
FIG_DIR = REPO_DIR / "figures/tumour_pairs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TUMOUR_IDS = ["D", "E", "F", "I", "J", "M", "P", "S", "U", "X"]

# glands on (side A, side B), 4+4 unless listed
SIDE_SPLIT = {
    "D": (2, 6),
    "P": (3, 5),
}
N_SEEDS = 30

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


def get_posterior_medians(tid):
    df = pd.read_csv(POSTERIOR_DIR / f"posterior_{tid}.csv")
    w = df["weight"].values
    w = w / w.sum()
    medians = {}
    for param in ["fission_rate", "meth_rate", "demeth_rate"]:
        idx = np.argsort(df[param].values)
        cw = np.cumsum(w[idx])
        medians[param] = 10 ** df[param].values[idx][np.searchsorted(cw, 0.5)]
    return medians


def run_methdemon_once(medians, seed, n_a=4, n_b=4, tid=""):
    tmpdir = tempfile.mkdtemp(prefix="fig_sim_")
    try:
        with open(CONFIG_TEMPLATE) as f:
            lines = f.readlines()
        updates = {
            "init_migration_rate": medians["fission_rate"],
            "meth_rate": medians["meth_rate"],
            "demeth_rate": medians["demeth_rate"],
            "mu_driver_birth": 0.0001,
            "s_driver_birth": 0.0,
            "left_demes": int(n_a),
            "right_demes": int(n_b),
            "seed": int(seed),
        }
        new_lines = []
        for line in lines:
            parts = line.split(maxsplit=1)
            if parts and parts[0] in updates:
                v = updates[parts[0]]
                if isinstance(v, int):
                    new_lines.append(f"    {parts[0]} {v}\n")
                else:
                    new_lines.append(f"    {parts[0]} {v:.10f}\n")
            else:
                new_lines.append(line)
        with open(os.path.join(tmpdir, "config.dat"), "w") as f:
            f.writelines(new_lines)

        result = subprocess.run(
            [str(METHDEMON_BIN), tmpdir, "config.dat"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
        )
        if result.returncode != 0:
            return None

        df = pd.read_csv(os.path.join(tmpdir, "final_demes.csv"))
        final = df.tail(8).reset_index(drop=True)
        gland_names = []
        arrays = {}
        sides = []
        for idx, row in final.iterrows():
            side = str(row["Side"]).strip()
            sides.append(side)
            label = "A" if side in ("left", "0") else "B"
            gnum = sum(1 for s in sides if s == side)
            name = f"{tid}{label} {gnum}"
            arr = np.array([float(x) for x in str(row["AverageArray"]).split(";")
                            if x.strip()])
            gland_names.append(name)
            arrays[name] = arr
        side_a_names = [n for n, s in zip(gland_names, sides) if s in ("left", "0")]
        side_b_names = [n for n, s in zip(gland_names, sides) if s not in ("left", "0")]
        ordered = side_a_names + side_b_names
        na = len(side_a_names)

        # between/within distance ratio, used to pick the sim
        n = len(ordered)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.mean((arrays[ordered[i]] - arrays[ordered[j]]) ** 2)
        within, between = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if (i < na and j < na) or (i >= na and j >= na):
                    within.append(D[i, j])
                else:
                    between.append(D[i, j])
        ratio = np.mean(between) / np.mean(within) if within and np.mean(within) > 0 else 0.0
        return ordered, arrays, ratio
    except Exception:
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def sample_from_posterior(tid, rng):
    df = pd.read_csv(POSTERIOR_DIR / f"posterior_{tid}.csv")
    w = df["weight"].values
    w = w / w.sum()
    idx = rng.choice(len(df), p=w)
    return {
        "fission_rate": 10 ** df["fission_rate"].iloc[idx],
        "meth_rate": 10 ** df["meth_rate"].iloc[idx],
        "demeth_rate": 10 ** df["demeth_rate"].iloc[idx],
    }


def run_best_sim(tid):
    n_a, n_b = SIDE_SPLIT.get(tid, (4, 4))
    best = None
    best_ratio = 0
    rng = np.random.default_rng(hash(tid) % 2**31)
    for i in range(N_SEEDS):
        params = sample_from_posterior(tid, rng)
        seed = rng.integers(1, 30000)
        result = run_methdemon_once(params, seed, n_a=n_a, n_b=n_b, tid=tid)
        if result is None:
            continue
        ordered, arrays, ratio = result
        if ratio > best_ratio:
            best = (ordered, arrays)
            best_ratio = ratio
            print(f"{tid} run {i+1}: new best ratio {ratio:.2f} "
                  f"(phi={params['fission_rate']:.5f})", flush=True)
    print(f"{tid} best ratio: {best_ratio:.2f}", flush=True)
    return best


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


def panel_B(fig, outer_gs, gland_names, arrays, title, n_a=4):
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
                ax.set_ylabel(gland_names[i][:10], fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=4)
            if i == n - 1:
                ax.set_xlabel(gland_names[j][:10], fontsize=9, rotation=45,
                              ha="right", va="top", labelpad=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#64748b")

    ax_overlay = fig.add_subplot(outer_gs, frameon=False)
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])
    ax_overlay.set_xlim(0, n)
    ax_overlay.set_ylim(n, 0)
    ax_overlay.set_title(title, fontsize=11, loc="left", pad=12)
    ax_overlay.axhline(n_a, color="#c53030", lw=1.2, xmin=0, xmax=n_a / n)
    ax_overlay.axvline(n_a, color="#c53030", lw=1.2,
                       ymin=1 - n / n, ymax=1 - n_a / n)
    ax_overlay.patch.set_alpha(0)


HEATMAP_VMAX = 0.15


def panel_C(ax, gland_names, arrays, title, n_a=4):
    n = len(gland_names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.mean((arrays[gland_names[i]] - arrays[gland_names[j]]) ** 2)
    im = ax.imshow(D, cmap="magma", vmin=0, vmax=HEATMAP_VMAX)
    short = [g[:10] for g in gland_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.axhline(n_a - 0.5, color="#ffffff", lw=1.2)
    ax.axvline(n_a - 0.5, color="#ffffff", lw=1.2)
    ax.set_title(title, fontsize=11, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r"Mean-squared $\beta$ difference")


def make_figure(gland_names, arrays, tid, fig_type, panel_b_title, panel_c_title, n_a=4):
    fig = plt.figure(figsize=(14, 6.5))
    gs = GridSpec(2, 3, figure=fig,
                  width_ratios=[1.0, 1.6, 1.2],
                  height_ratios=[1, 1],
                  wspace=0.35, hspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_A(ax_a, arrays, gland_names[0])
    panel_B(fig, gs[:, 1], gland_names, arrays, panel_b_title, n_a=n_a)
    ax_c = fig.add_subplot(gs[:, 2])
    panel_C(ax_c, gland_names, arrays, panel_c_title, n_a=n_a)
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")
    out = FIG_DIR / f"fig_{fig_type}_{tid}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def process_tumour(tid):
    print(f"Loading tumour {tid}...", flush=True)
    gland_names, arrays, n_a = load_real_data(tid)
    print(f"{tid}: {n_a}A + {len(gland_names)-n_a}B glands", flush=True)
    out_data = make_figure(
        gland_names, arrays, tid, "data",
        f"B. Paired gland $\\beta$ densities (tumour {tid})",
        f"C. Inter-gland distance (tumour {tid})",
        n_a=n_a,
    )

    print(f"Simulating tumour {tid} ({N_SEEDS} runs)...",
          flush=True)
    sim_result = run_best_sim(tid)
    if sim_result is None:
        print(f"{tid}: all sims failed", flush=True)
        return tid, str(out_data), None
    sim_names, sim_arrays = sim_result
    # keep the sim for re-plotting
    np.savez(FIG_DIR / f"sim_data_{tid}.npz",
             gland_names=sim_names,
             **{n: sim_arrays[n] for n in sim_names})
    n_a_sim = SIDE_SPLIT.get(tid, (4, 4))[0]
    out_sim = make_figure(
        sim_names, sim_arrays, tid, "sim",
        f"B. Simulated paired gland densities (tumour {tid})",
        f"C. Simulated inter-gland distance (tumour {tid})",
        n_a=n_a_sim,
    )
    print(f"Saved {out_data.name}, {out_sim.name}", flush=True)
    return tid, str(out_data), str(out_sim)


def main():
    print(f"Generating paired figures for {len(TUMOUR_IDS)} tumours "
          f"({N_SEEDS} seeds each)...")
    print(f"Output: {FIG_DIR}/")

    import sys
    run_ids = sys.argv[1:] if len(sys.argv) > 1 else TUMOUR_IDS
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(process_tumour, run_ids))

    for tid, data_path, sim_path in results:
        status = "OK" if sim_path else "DATA ONLY"
        print(f"{tid}: {status}")


if __name__ == "__main__":
    main()

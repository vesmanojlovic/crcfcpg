#!/usr/bin/env python3
"""Simulated version of the data figure, using tumour U posterior medians."""

import os
import shutil
import subprocess
import tempfile
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
METHDEMON_BIN = METHABC_DIR / "resources/methdemon/bin/methdemon"
CONFIG_TEMPLATE = METHABC_DIR / "resources/config_template.dat"
POSTERIOR_DIR = REPO_DIR / "outputs"
FIG_DIR = REPO_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

YELLOW_TO_PURPLE = LinearSegmentedColormap.from_list(
    "yellow_purple", ["#1e1b4b", "#4c1d95", "#7c3aed", "#fbbf24", "#fef9c3"]
)


def get_tumour_U_medians():
    df = pd.read_csv(POSTERIOR_DIR / "posterior_U.csv")
    w = df["weight"].values
    w = w / w.sum()
    medians = {}
    for param in ["fission_rate", "meth_rate", "demeth_rate"]:
        idx = np.argsort(df[param].values)
        cw = np.cumsum(w[idx])
        medians[param] = 10 ** df[param].values[idx][np.searchsorted(cw, 0.5)]
    return medians


def write_config(out_dir, medians):
    with open(CONFIG_TEMPLATE) as f:
        lines = f.readlines()
    updates = {
        "init_migration_rate": medians["fission_rate"],
        "meth_rate": medians["meth_rate"],
        "demeth_rate": medians["demeth_rate"],
        "mu_driver_birth": 0.0001,
        "s_driver_birth": 0.0,
        "seed": int(42),
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
    cfg_path = os.path.join(out_dir, "config.dat")
    with open(cfg_path, "w") as f:
        f.writelines(new_lines)
    return cfg_path


def run_methdemon(medians):
    tmpdir = tempfile.mkdtemp(prefix="modelfig_")
    try:
        write_config(tmpdir, medians)
        result = subprocess.run(
            [str(METHDEMON_BIN), tmpdir, "config.dat"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"methdemon failed: {result.stderr.decode()[:500]}")
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
            name = f"S{label} {gnum}"
            arr = np.array([float(x) for x in str(row["AverageArray"]).split(";")
                            if x.strip()])
            gland_names.append(name)
            arrays[name] = arr
        side_a = [n for n, s in zip(gland_names, sides)
                  if s in ("left", "0")]
        side_b = [n for n, s in zip(gland_names, sides)
                  if s not in ("left", "0")]
        ordered = side_a + side_b
        return ordered, arrays
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def panel_A(ax, arrays, gland):
    beta = arrays[gland]
    ax.hist(beta, bins=40, range=(0, 1), color="#2563eb",
            edgecolor="white", linewidth=0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"fCpG $\beta$")
    ax.set_ylabel("Count")
    ax.set_title(f"A. Single simulated gland", fontsize=11, loc="left")
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
                label = gland_names[i].replace("Side ", "S").replace(", gland ", "")
                ax.set_ylabel(label, fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=4)
            if i == n - 1:
                label = gland_names[j].replace("Side ", "S").replace(", gland ", "")
                ax.set_xlabel(label, fontsize=7, rotation=45,
                              ha="right", va="top", labelpad=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#64748b")

    ax_overlay = fig.add_subplot(outer_gs, frameon=False)
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])
    ax_overlay.set_xlim(0, n)
    ax_overlay.set_ylim(n, 0)
    ax_overlay.set_title("B. Simulated paired gland densities",
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
    short = [g.replace("Side ", "S").replace(", gland ", "")
             for g in gland_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.axhline(3.5, color="#ffffff", lw=1.2)
    ax.axvline(3.5, color="#ffffff", lw=1.2)
    ax.set_title("C. Simulated inter-gland distance", fontsize=11, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean sq. diff.")


def main():
    print("Getting tumour U posterior medians...")
    medians = get_tumour_U_medians()
    print(f"  fission={medians['fission_rate']:.6f}, "
          f"meth={medians['meth_rate']:.6f}, "
          f"demeth={medians['demeth_rate']:.6f}")

    print("Running methdemon...")
    gland_names, arrays = run_methdemon(medians)
    print(f"  {len(gland_names)} glands, "
          f"{len(next(iter(arrays.values())))} fCpG loci")
    print(f"  Order: {gland_names}")

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

    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.axis("off")

    out_pdf = FIG_DIR / "fig_model_data.pdf"
    out_png = FIG_DIR / "fig_model_data.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

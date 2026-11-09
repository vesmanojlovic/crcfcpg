#!/usr/bin/env python3
"""Illustrative methdemon runs showing how epimutation and fission rates shape the output."""

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
import numpy as np
import pandas as pd
from methabc.utils import repo_dir

REPO_DIR = Path(__file__).resolve().parents[1]
METHABC_DIR = Path(repo_dir)
METHDEMON_BIN = METHABC_DIR / "resources/methdemon/bin/methdemon"
CONFIG_TEMPLATE = METHABC_DIR / "resources/config_template.dat"
FIGURE_DIR = REPO_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def write_config(out_dir, updates):
    with open(CONFIG_TEMPLATE) as f:
        lines = f.readlines()
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


def run_sim(params):
    """Run methdemon, returning (all betas concatenated, distance matrix, per-gland arrays)."""
    tmpdir = tempfile.mkdtemp(prefix="paramsens_")
    try:
        write_config(tmpdir, params)
        result = subprocess.run(
            [str(METHDEMON_BIN), tmpdir, "config.dat"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"methdemon failed: {result.stderr.decode()[:300]}")
        df = pd.read_csv(os.path.join(tmpdir, "final_demes.csv"))
        final = df.tail(8).reset_index(drop=True)
        arrays = []
        for _, row in final.iterrows():
            arr = np.array([float(x) for x in str(row["AverageArray"]).split(";")
                            if x.strip()])
            arrays.append(arr)
        n = len(arrays)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.mean((arrays[i] - arrays[j]) ** 2)
        return np.concatenate(arrays), D, arrays
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_scenario(args):
    name, params = args
    betas, D, arrays = run_sim(params)
    print(f"{name} done: mean beta={betas.mean():.3f}, "
          f"mean D={D[np.triu_indices(len(arrays), k=1)].mean():.4f}", flush=True)
    return name, betas, D, arrays


def main():
    BASE = {
        "deme_carrying_capacity": 100,
        "init_migration_rate":    0.003,
        "mu_driver_birth":        0.0001,
        "s_driver_birth":         0.0,
        "meth_rate":              0.001,
        "demeth_rate":            0.001,
        "fCpG_loci_per_cell":     1200,
        "max_fissions":           23,
        "turnover":               0.3,
        "seed":                   12345,
    }

    def cfg(**over):
        out = dict(BASE)
        out.update(over)
        return out

    scenarios = [
        ("low_epi",    cfg(meth_rate=3e-5,  demeth_rate=3e-5)),
        ("mid_epi",    cfg(meth_rate=1e-3,  demeth_rate=1e-3)),
        ("high_epi",   cfg(meth_rate=2e-2,  demeth_rate=2e-2)),
        ("biased_epi", cfg(meth_rate=5e-3,  demeth_rate=5e-4)),
        ("low_fission",  cfg(init_migration_rate=5e-4, seed=777)),
        ("high_fission", cfg(init_migration_rate=5e-2, seed=777)),
    ]

    print("Running 6 illustrative methdemon simulations in parallel...")
    results = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for name, betas, D, arrays in pool.map(run_scenario, scenarios):
            results[name] = (betas, D, arrays)

    fig = plt.figure(figsize=(15, 7.5))
    gs = GridSpec(2, 4, figure=fig,
                  width_ratios=[1, 1, 1, 1],
                  hspace=0.55, wspace=0.55)

    # top row: epimutation rate sweep
    epi_labels = [
        ("low_epi",    "Low epimutation"),
        ("mid_epi",    "Intermediate epimutation"),
        ("high_epi",   "High epimutation"),
        ("biased_epi", "Asymmetric\nmethylation / demethylation"),
    ]
    for col, (key, title) in enumerate(epi_labels):
        ax = fig.add_subplot(gs[0, col])
        betas, _, _ = results[key]
        ax.hist(betas, bins=40, range=(0, 1), color="#2b6cb0",
                edgecolor="white", linewidth=0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$\beta$")
        if col == 0:
            ax.set_ylabel("Count (fCpG sites × glands)")
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # bottom left: fission rate
    fiss_labels = [
        ("low_fission",  "Low fission rate"),
        ("high_fission", "High fission rate"),
    ]
    _, D_lo, _ = results["low_fission"]
    _, D_hi, _ = results["high_fission"]
    vmax = max(D_lo.max(), D_hi.max())

    for col, (key, title) in enumerate(fiss_labels):
        ax = fig.add_subplot(gs[1, col])
        _, D, _ = results[key]
        im = ax.imshow(D, vmin=0, vmax=vmax, cmap="magma")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Gland")
        if col == 0:
            ax.set_ylabel("Gland")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # bottom right: single-gland examples of the bimodal, W and unimodal shapes
    ax = fig.add_subplot(gs[1, 2])
    _, _, arr_mid = results["mid_epi"]
    _, _, arr_lo = results["low_epi"]
    ax.hist(arr_lo[0], bins=30, range=(0, 1), alpha=0.6,
            color="#2b6cb0", label="Low flip\nrates", edgecolor="white", linewidth=0.3)
    ax.hist(arr_mid[0], bins=30, range=(0, 1), alpha=0.6,
            color="#dd6b20", label="Intermediate\nflip rates", edgecolor="white",
            linewidth=0.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$\beta$ (single gland)")
    ax.set_ylabel("Count")
    ax.set_title("W-shape", fontsize=11)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(gs[1, 3])
    _, _, arr_hi = results["high_epi"]
    _, _, arr_bias = results["biased_epi"]
    ax.hist(arr_hi[0], bins=30, range=(0, 1), alpha=0.6,
            color="#2b6cb0", label="Symmetric high\nflip rates",
            edgecolor="white", linewidth=0.3)
    ax.hist(arr_bias[0], bins=30, range=(0, 1), alpha=0.6,
            color="#c53030", label="Asymmetric\n(meth ≫ demeth)",
            edgecolor="white", linewidth=0.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$\beta$ (single gland)")
    ax.set_ylabel("Count")
    ax.set_title("Unimodal peak location", fontsize=11)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_pdf = FIGURE_DIR / "sfig_parameter_sensitivity.pdf"
    out_png = FIGURE_DIR / "sfig_parameter_sensitivity.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

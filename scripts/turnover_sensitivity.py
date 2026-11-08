#!/usr/bin/env python3
"""Sensitivity of methdemon output to the turnover fraction, over a Latin hypercube of parameters."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from methabc.utils import repo_dir

REPO_DIR = Path(__file__).resolve().parents[1]
METHABC_DIR = Path(repo_dir)
METHDEMON_BIN = METHABC_DIR / "resources/methdemon/bin/methdemon"
CONFIG_TEMPLATE = METHABC_DIR / "resources/config_template.dat"
OUTPUT_DIR = REPO_DIR / "outputs/turnover_sensitivity"
FIGURE_DIR = REPO_DIR / "figures"

TURNOVER_FRACTIONS = [0.0, 0.15, 0.3, 0.5]

# log10 prior ranges, as in inference
PRIOR_RANGES = {
    "init_migration_rate": (-3.3, -1.0),
    "meth_rate":           (-4.0, -2.0),
    "demeth_rate":         (-4.0, -2.0),
    "mu_driver_birth":     (-5.0, -2.0),
    "s_driver_birth":      ( 0.0,  0.2),
}


def latin_hypercube(n, ranges, rng):
    d = len(ranges)
    keys = list(ranges.keys())
    samples = np.zeros((n, d))
    for j in range(d):
        lo, hi = ranges[keys[j]]
        perm = rng.permutation(n)
        for i in range(n):
            samples[i, j] = lo + (perm[i] + rng.random()) / n * (hi - lo)
    return [{k: samples[i, j] for j, k in enumerate(keys)}
            for i in range(n)]


def write_config(log_params, turnover, seed, output_dir):
    # s_driver_birth is not on log scale
    nat = {}
    for k, v in log_params.items():
        nat[k] = 10**v if k != "s_driver_birth" else v

    param_updates = {
        "init_migration_rate": nat["init_migration_rate"],
        "meth_rate":           nat["meth_rate"],
        "demeth_rate":         nat["demeth_rate"],
        "mu_driver_birth":     nat["mu_driver_birth"],
        "s_driver_birth":      nat["s_driver_birth"],
        "mu_driver_migration": 1e-10,
        "s_driver_migration":  1e-10,
        "turnover":            turnover,
        "seed":                int(seed),
    }

    with open(CONFIG_TEMPLATE, "r") as f:
        lines = f.readlines()

    out = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if parts and parts[0] in param_updates:
            val = param_updates[parts[0]]
            if isinstance(val, (int, np.integer)):
                out.append(f"    {parts[0]} {val}\n")
            else:
                out.append(f"    {parts[0]} {val:.10f}\n")
        else:
            out.append(line)

    path = os.path.join(output_dir, "config.dat")
    with open(path, "w") as f:
        f.writelines(out)
    return path


def run_single(log_params, turnover, seed):
    """Run methdemon once, returning summary statistics or None on failure."""
    tmpdir = tempfile.mkdtemp(prefix="tsens_")
    try:
        write_config(log_params, turnover, seed, tmpdir)
        result = subprocess.run(
            [str(METHDEMON_BIN), tmpdir, "config.dat"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=300,
        )
        if result.returncode != 0:
            return None

        csv_path = os.path.join(tmpdir, "final_demes.csv")
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path)
        if len(df) < 8:
            return None

        final = df.tail(8).reset_index(drop=True)
        arrays, sides = [], []
        for _, row in final.iterrows():
            arr = np.array([float(x) for x in
                            str(row["AverageArray"]).split(";") if x.strip()])
            arrays.append(arr)
            sides.append(row["Side"])

        if len(arrays) != 8:
            return None

        n = 8
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.mean((arrays[i] - arrays[j])**2)

        within, between = [], []
        for i in range(n):
            for j in range(i + 1, n):
                (within if sides[i] == sides[j] else between).append(D[i, j])

        return {
            "dist_matrix": D,
            "mean_all":     float(np.mean(D[np.triu_indices(n, k=1)])),
            "mean_within":  float(np.mean(within)) if within else np.nan,
            "mean_between": float(np.mean(between)) if between else np.nan,
            "ratio":        (float(np.mean(between) / np.mean(within))
                             if within and np.mean(within) > 0 else np.nan),
        }
    except Exception:
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_fraction(args):
    """Run all parameter sets at one turnover fraction."""
    turnover, param_list, seeds = args
    results = []
    n = len(param_list)
    for i, (p, s) in enumerate(zip(param_list, seeds)):
        r = run_single(p, turnover, s)
        results.append(r)
        if (i + 1) % 25 == 0:
            ok = sum(1 for x in results if x is not None)
            print(f"turnover {turnover:.2f}: {i+1}/{n} done "
                  f"({ok} successful)", flush=True)
    ok = sum(1 for x in results if x is not None)
    print(f"turnover {turnover:.2f}: finished, {ok}/{n} successful",
          flush=True)
    return turnover, results


def analyse(all_results, param_list):
    fractions = sorted(all_results.keys())
    n_params = len(param_list)

    # only keep parameter sets that succeeded at every fraction
    valid_idx = []
    for i in range(n_params):
        if all(all_results[f][i] is not None for f in fractions):
            valid_idx.append(i)

    print(f"Valid parameter sets: "
          f"{len(valid_idx)}/{n_params}")

    metrics = {}
    for f in fractions:
        metrics[f] = {
            "mean_all":    [all_results[f][i]["mean_all"] for i in valid_idx],
            "mean_within": [all_results[f][i]["mean_within"] for i in valid_idx],
            "mean_between":[all_results[f][i]["mean_between"] for i in valid_idx],
            "ratio":       [all_results[f][i]["ratio"] for i in valid_idx],
        }

    # frobenius distance between distance matrices from the same params
    frob = {}
    baseline = 0.3
    for f in fractions:
        if f == baseline:
            continue
        diffs = []
        for i in valid_idx:
            D_base = all_results[baseline][i]["dist_matrix"]
            D_f = all_results[f][i]["dist_matrix"]
            frob_dist = np.sqrt(np.sum((D_base - D_f)**2))
            diffs.append(frob_dist)
        frob[f] = diffs
        median_frob = np.median(diffs)
        baseline_mag = np.median(metrics[baseline]["mean_all"])
        print(f"Frobenius 30% vs {f:.0%}: median={median_frob:.4f} "
              f"(baseline mean_all median={baseline_mag:.4f}, "
              f"ratio={median_frob/baseline_mag:.2f})")

    print("Wilcoxon signed-rank tests, mean_all vs 30% baseline:")
    for f in fractions:
        if f == baseline:
            continue
        base_vals = np.array(metrics[baseline]["mean_all"])
        test_vals = np.array(metrics[f]["mean_all"])
        stat, p = stats.wilcoxon(base_vals, test_vals)
        diff = np.median(test_vals - base_vals)
        print(f"  {f:.0%} vs 30%: median_diff={diff:+.5f}, "
              f"W={stat:.0f}, p={p:.4f}")

    return metrics, frob, valid_idx


def make_plots(metrics, frob, fractions):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    ax = axes[0]
    positions = range(len(fractions))
    data = [metrics[f]["mean_all"] for f in fractions]
    vp = ax.violinplot(data, positions=positions, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:.0%}" for f in fractions])
    ax.set_xlabel("Turnover fraction")
    ax.set_ylabel("Mean pairwise distance")
    ax.set_title("A. Mean inter-gland distance")

    ax = axes[1]
    data = [metrics[f]["ratio"] for f in fractions]
    vp = ax.violinplot(data, positions=positions, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:.0%}" for f in fractions])
    ax.set_xlabel("Turnover fraction")
    ax.set_ylabel("Between / within distance ratio")
    ax.set_title("B. Block structure preservation")

    ax = axes[2]
    other_fracs = [f for f in fractions if f != 0.3]
    data = [frob[f] for f in other_fracs]
    vp = ax.violinplot(data, positions=range(len(other_fracs)),
                       showmedians=True)
    ax.set_xticks(range(len(other_fracs)))
    ax.set_xticklabels([f"{f:.0%}" for f in other_fracs])
    ax.set_xlabel("Turnover fraction")
    ax.set_ylabel("Frobenius distance from 30% baseline")
    ax.set_title("C. Distance matrix shift from baseline")

    fig.suptitle("Turnover sensitivity analysis", fontsize=14)
    fig.tight_layout()

    out = FIGURE_DIR / "sfig_turnover_sensitivity.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Turnover sensitivity: grid search across parameter "
                    "space at multiple turnover fractions"
    )
    parser.add_argument("-n", "--n-params", type=int, default=200,
                        help="Number of parameter sets (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for Latin hypercube (default: 42)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    n = args.n_params
    rng = np.random.default_rng(args.seed)

    print(f"Running {n} parameter sets at turnover fractions {TURNOVER_FRACTIONS}")

    param_list = latin_hypercube(n, PRIOR_RANGES, rng)

    # same seed for a parameter set at every fraction
    seeds = rng.integers(1, 2**15 - 1, size=n).tolist()

    # one process per fraction
    jobs = [(f, param_list, seeds) for f in TURNOVER_FRACTIONS]

    all_results = {}
    with ProcessPoolExecutor(max_workers=len(TURNOVER_FRACTIONS)) as pool:
        futures = {pool.submit(run_fraction, j): j[0] for j in jobs}
        for future in as_completed(futures):
            turnover, results = future.result()
            all_results[turnover] = results

    fractions = sorted(all_results.keys())
    metrics, frob, valid_idx = analyse(all_results, param_list)

    make_plots(metrics, frob, fractions)

    summary = {
        "n_params": n,
        "n_valid": len(valid_idx),
        "fractions": TURNOVER_FRACTIONS,
        "medians": {
            f"{f:.2f}": {
                "mean_all": float(np.median(metrics[f]["mean_all"])),
                "mean_within": float(np.median(metrics[f]["mean_within"])),
                "mean_between": float(np.median(metrics[f]["mean_between"])),
                "ratio": float(np.median(metrics[f]["ratio"])),
            }
            for f in fractions
        },
        "frobenius_vs_baseline": {
            f"{f:.2f}": {
                "median": float(np.median(frob[f])),
                "mean": float(np.mean(frob[f])),
            }
            for f in frob
        },
    }
    out = OUTPUT_DIR / "sensitivity_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

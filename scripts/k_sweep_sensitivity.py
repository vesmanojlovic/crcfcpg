#!/usr/bin/env python3
"""Sensitivity of simulated inter-gland distances to the deme carrying capacity K."""

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
DATA_DIR = REPO_DIR / "data/individual"
OUTPUT_DIR = REPO_DIR / "outputs/k_sweep"
FIGURE_DIR = REPO_DIR / "figures"

K_VALUES = [10, 50, 100, 500, 1000]

# prior ranges on log10 scale, as in inference
PRIOR_RANGES = {
    "init_migration_rate": (-3.3, -1.0),
    "meth_rate":           (-4.0, -2.0),
    "demeth_rate":         (-4.0, -2.0),
    "mu_driver_birth":     (-5.0, -2.0),
    "s_driver_birth":      ( 0.0,  0.2),
}

# observed distances for reference lines, used if present
OBSERVED_DISTANCES_FILE = REPO_DIR / "outputs/cohort_statistics.json"


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


def write_config(log_params, K, seed, output_dir):
    # log10 -> natural scale, except s_driver_birth
    nat = {}
    for k, v in log_params.items():
        nat[k] = 10**v if k != "s_driver_birth" else v

    param_updates = {
        "deme_carrying_capacity": int(K),
        "init_migration_rate": nat["init_migration_rate"],
        "meth_rate":           nat["meth_rate"],
        "demeth_rate":         nat["demeth_rate"],
        "mu_driver_birth":     nat["mu_driver_birth"],
        "s_driver_birth":      nat["s_driver_birth"],
        "mu_driver_migration": 1e-10,
        "s_driver_migration":  1e-10,
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


def run_single(log_params, K, seed):
    """Run methdemon once, return summary dict or None on failure."""
    tmpdir = tempfile.mkdtemp(prefix="ksweep_")
    try:
        write_config(log_params, K, seed, tmpdir)
        result = subprocess.run(
            [str(METHDEMON_BIN), tmpdir, "config.dat"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=600,  # higher K may be slower
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


def run_K_value(args):
    """Run all parameter sets at one K (one process per K)."""
    K, param_list, seeds = args
    results = []
    n = len(param_list)
    for i, (p, s) in enumerate(zip(param_list, seeds)):
        r = run_single(p, K, s)
        results.append(r)
        if (i + 1) % 25 == 0:
            ok = sum(1 for x in results if x is not None)
            print(f"[K={K}] {i+1}/{n} done "
                  f"({ok} successful)", flush=True)
    ok = sum(1 for x in results if x is not None)
    print(f"[K={K}] finished, {ok}/{n} successful", flush=True)
    return K, results


def load_observed_distances():
    if OBSERVED_DISTANCES_FILE.exists():
        with open(OBSERVED_DISTANCES_FILE) as f:
            data = json.load(f)
        if "mean_pairwise_distance" in data:
            return data["mean_pairwise_distance"]
    return None


def analyse(all_results, param_list):
    k_vals = sorted(all_results.keys())
    n_params = len(param_list)

    # only keep param sets that succeeded at every K
    valid_idx = []
    for i in range(n_params):
        if all(all_results[k][i] is not None for k in k_vals):
            valid_idx.append(i)

    print(f"Valid parameter sets (succeeded at all K): "
          f"{len(valid_idx)}/{n_params}")

    metrics = {}
    for k in k_vals:
        metrics[k] = {
            "mean_all":    [all_results[k][i]["mean_all"] for i in valid_idx],
            "mean_within": [all_results[k][i]["mean_within"] for i in valid_idx],
            "mean_between":[all_results[k][i]["mean_between"] for i in valid_idx],
            "ratio":       [all_results[k][i]["ratio"] for i in valid_idx],
        }

    print("Median mean pairwise distance by K:")
    for k in k_vals:
        med = np.median(metrics[k]["mean_all"])
        q25 = np.percentile(metrics[k]["mean_all"], 25)
        q75 = np.percentile(metrics[k]["mean_all"], 75)
        print(f"  K={k:>5d}: median={med:.5f}  (IQR: {q25:.5f}-{q75:.5f})")

    frob = {}
    baseline = 100
    for k in k_vals:
        if k == baseline:
            continue
        diffs = []
        for i in valid_idx:
            D_base = all_results[baseline][i]["dist_matrix"]
            D_k = all_results[k][i]["dist_matrix"]
            frob_dist = np.sqrt(np.sum((D_base - D_k)**2))
            diffs.append(frob_dist)
        frob[k] = diffs
        median_frob = np.median(diffs)
        baseline_mag = np.median(metrics[baseline]["mean_all"])
        print(f"Frobenius(K=100 vs K={k}): median={median_frob:.5f} "
              f"(baseline mean_all median={baseline_mag:.5f}, "
              f"ratio={median_frob/baseline_mag:.2f})")

    return metrics, frob, valid_idx


def make_plots(metrics, frob, k_vals, obs_distances=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    positions = range(len(k_vals))
    labels = [str(k) for k in k_vals]

    ax = axes[0]
    data = [metrics[k]["mean_all"] for k in k_vals]
    vp = ax.violinplot(data, positions=positions, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Carrying capacity K")
    ax.set_ylabel("Mean pairwise distance")
    ax.set_title("A. Mean inter-gland distance")
    ax.set_yscale("log")
    if obs_distances is not None:
        if isinstance(obs_distances, dict):
            vals = list(obs_distances.values())
        elif isinstance(obs_distances, (list, np.ndarray)):
            vals = list(obs_distances)
        else:
            vals = None
        if vals:
            ax.axhspan(min(vals), max(vals), alpha=0.2, color="green",
                       label="Observed range")
            ax.legend(fontsize=8)

    ax = axes[1]
    data = [metrics[k]["ratio"] for k in k_vals]
    vp = ax.violinplot(data, positions=positions, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Carrying capacity K")
    ax.set_ylabel("Between / within distance ratio")
    ax.set_title("B. Block structure preservation")

    ax = axes[2]
    other_k = [k for k in k_vals if k != 100]
    data = [frob[k] for k in other_k]
    vp = ax.violinplot(data, positions=range(len(other_k)),
                       showmedians=True)
    ax.set_xticks(range(len(other_k)))
    ax.set_xticklabels([str(k) for k in other_k])
    ax.set_xlabel("Carrying capacity K")
    ax.set_ylabel("Frobenius distance from K=100")
    ax.set_title("C. Distance matrix shift from K=100")

    fig.suptitle("Carrying capacity sensitivity analysis", fontsize=14)
    fig.tight_layout()

    out = FIGURE_DIR / "sfig_k_sweep.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Carrying capacity sensitivity: grid search across "
                    "parameter space at multiple K values"
    )
    parser.add_argument("-n", "--n-params", type=int, default=200,
                        help="Number of parameter sets (default: 200)")
    parser.add_argument("--seed", type=int, default=123,
                        help="RNG seed for Latin hypercube (default: 123)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    n = args.n_params
    rng = np.random.default_rng(args.seed)

    print(f"{n} parameter sets x K={K_VALUES} ({n * len(K_VALUES)} sims)")

    param_list = latin_hypercube(n, PRIOR_RANGES, rng)

    # same seed for a param set at every K, so only K differs
    seeds = rng.integers(1, 2**15 - 1, size=n).tolist()

    jobs = [(k, param_list, seeds) for k in K_VALUES]

    all_results = {}
    with ProcessPoolExecutor(max_workers=len(K_VALUES)) as pool:
        futures = {pool.submit(run_K_value, j): j[0] for j in jobs}
        for future in as_completed(futures):
            K, results = future.result()
            all_results[K] = results

    k_vals = sorted(all_results.keys())
    metrics, frob, valid_idx = analyse(all_results, param_list)

    obs_distances = load_observed_distances()

    make_plots(metrics, frob, k_vals, obs_distances)

    summary = {
        "n_params": n,
        "n_valid": len(valid_idx),
        "k_values": K_VALUES,
        "medians": {
            str(k): {
                "mean_all": float(np.median(metrics[k]["mean_all"])),
                "mean_within": float(np.median(metrics[k]["mean_within"])),
                "mean_between": float(np.median(metrics[k]["mean_between"])),
                "ratio": float(np.median(metrics[k]["ratio"])),
            }
            for k in k_vals
        },
        "frobenius_vs_K100": {
            str(k): {
                "median": float(np.median(frob[k])),
                "mean": float(np.mean(frob[k])),
            }
            for k in frob
        },
    }
    out = OUTPUT_DIR / "k_sweep_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

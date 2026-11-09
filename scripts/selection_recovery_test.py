#!/usr/bin/env python3
"""
Synthetic test of whether a polarisation heterogeneity statistic makes mu and s
identifiable: infer from simulated data with and without it.
"""

import multiprocessing
# must be set before any other multiprocessing use; forkserver can't pickle
# the pyabc closures
try:
    multiprocessing.set_start_method('fork')
except RuntimeError:
    pass

import os
import sys
import subprocess
import tempfile
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

import pyabc
from methabc.utils import repo_dir

REPO_DIR = Path(__file__).resolve().parents[1]
METHABC_DIR = Path(repo_dir)
METHDEMON_BIN = METHABC_DIR / "resources" / "methdemon" / "bin" / "methdemon"
CONFIG_TEMPLATE = METHABC_DIR / "resources" / "config_template.dat"
FIGURE_DIR = REPO_DIR / "figures"
OUTPUT_DIR = REPO_DIR / "outputs"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH = {
    'meth_rate': 0.0022,
    'demeth_rate': 0.0018,
    'init_migration_rate': 0.009,
    'mu_driver_birth': 0.0001,
    's_driver_birth': 0.1,
    'deme_carrying_capacity': 100,
}

GROUND_TRUTH_LOG = {
    'meth_rate': np.log10(0.0022),
    'demeth_rate': np.log10(0.0018),
    'init_migration_rate': np.log10(0.009),
    'mu_driver_birth': np.log10(0.0001),
    's_driver_birth': 0.1,
}

# same priors as the real inference
LOG_PRIORS = {
    'meth_rate': (-4, -2),
    'demeth_rate': (-4, -2),
    'init_migration_rate': (-3.3, -1),
    'mu_driver_birth': (-5, -2),
    's_driver_birth': (0, 0.2),
}


def write_config(params, output_dir):
    with open(CONFIG_TEMPLATE, 'r') as f:
        lines = f.readlines()

    seed = np.random.randint(1, 2**15 - 1)
    params_with_seed = dict(params)
    params_with_seed['seed'] = seed

    updated_lines = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if parts and parts[0] in params_with_seed:
            val = params_with_seed[parts[0]]
            if isinstance(val, (int, np.integer)):
                updated_lines.append(f"    {parts[0]} {val}\n")
            else:
                updated_lines.append(f"    {parts[0]} {val:.10f}\n")
        else:
            updated_lines.append(line)

    config_path = os.path.join(output_dir, 'config.dat')
    with open(config_path, 'w') as f:
        f.writelines(updated_lines)

    return config_path


def simulate(params):
    """Run methdemon once and return the final-generation demes, or None on failure."""
    tmpdir = tempfile.mkdtemp(prefix='sel_test_')

    try:
        config_path = write_config(params, tmpdir)
        result = subprocess.run(
            [str(METHDEMON_BIN), tmpdir, 'config.dat'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )

        if result.returncode != 0:
            return None

        data_path = os.path.join(tmpdir, 'final_demes.csv')
        if not os.path.exists(data_path):
            return None

        df = pd.read_csv(data_path, header=0)
        df.AverageArray = df.AverageArray.apply(
            lambda x: np.fromstring(x, sep=";")
        )
        df = df[np.floor(df.Generation) == np.floor(df.Generation.max())]
        df = df.reset_index(drop=True)

        return df

    except Exception as e:
        return None
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def simulate_abc_log(log_params):
    """pyabc model taking log10 parameters (s_driver_birth on natural scale)."""
    params = {k: 10**v for k, v in log_params.items() if k != 's_driver_birth'}
    params['s_driver_birth'] = log_params['s_driver_birth']
    params['deme_carrying_capacity'] = 100
    res = simulate(params)
    return {"data": res}


def compute_deme_matrix(df):
    if df is None or len(df) != 8:
        return 100 * np.ones((8, 8))

    df = df.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
    res = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            arr1 = df.iloc[i].AverageArray
            arr2 = df.iloc[j].AverageArray
            diff = (arr1 - arr2) ** 2
            res[i, j] = np.sum(diff) / len(diff)
    return res


def l2_distance(dict1, dict2):
    """Frobenius norm of the difference of the gland distance matrices."""
    df1 = dict1['data']
    df2 = dict2['data']
    if df1 is None or df2 is None:
        return 1000.0
    x = compute_deme_matrix(df1)
    y = compute_deme_matrix(df2)
    diff = x - y
    return float(np.sqrt(np.sum(diff ** 2) / 2))


def overall_wasserstein(dict1, dict2):
    """Sum of per-gland Wasserstein distances."""
    df1 = dict1['data']
    df2 = dict2['data']
    if df1 is None or df2 is None:
        return 1000.0
    if len(df1) != 8 or len(df2) != 8:
        return 1000.0

    df1 = df1.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
    df2 = df2.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)

    res = 0
    for i in range(8):
        res += wasserstein_distance(
            df1.iloc[i].AverageArray,
            df2.iloc[i].AverageArray,
        )
    return float(res)


def polarisation_heterogeneity(dict1, dict2):
    """
    Absolute difference in the variance across glands of the polarisation
    P_i = mean(|2*beta - 1|). Sweeps in individual glands should raise it.
    """
    def get_gland_polarisations(d):
        df = d['data']
        if df is None or len(df) != 8:
            return None
        df = df.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
        pols = []
        for _, row in df.iterrows():
            arr = row['AverageArray']
            pol = np.mean(np.abs(2 * arr - 1))
            pols.append(pol)
        return np.array(pols)

    pols1 = get_gland_polarisations(dict1)
    pols2 = get_gland_polarisations(dict2)

    if pols1 is None or pols2 is None:
        return 1000.0

    var1 = np.var(pols1)
    var2 = np.var(pols2)

    return float(abs(var1 - var2))


def polarisation_profile_distance(dict1, dict2):
    """L2 distance between the sorted per-gland polarisation vectors."""
    def get_sorted_polarisations(d):
        df = d['data']
        if df is None or len(df) != 8:
            return None
        pols = []
        for _, row in df.iterrows():
            arr = row['AverageArray']
            pol = np.mean(np.abs(2 * arr - 1))
            pols.append(pol)
        return np.sort(pols)

    pols1 = get_sorted_polarisations(dict1)
    pols2 = get_sorted_polarisations(dict2)

    if pols1 is None or pols2 is None:
        return 1000.0

    return float(np.sqrt(np.sum((pols1 - pols2) ** 2)))


def run_inference(name, distance_fns, observed, db_path, n_cores=16,
                  pop_size=500, max_populations=12):
    print(f"Running {name}: population {pop_size}, "
          f"max {max_populations} generations, {n_cores} cores")
    print(f"  distances: {[f.__name__ for f in distance_fns]}")

    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a)
           for key, (a, b) in LOG_PRIORS.items()},
    )

    distance = pyabc.AdaptiveAggregatedDistance(distance_fns)

    sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_cores)

    abc = pyabc.ABCSMC(
        simulate_abc_log,
        prior,
        distance,
        population_size=pop_size,
        sampler=sampler,
    )

    abc_id = abc.new(
        db="sqlite:///" + str(db_path),
        observed_sum_stat={"data": observed},
        gt_par=GROUND_TRUTH_LOG,
    )

    t0 = time.time()
    history = abc.run(
        max_nr_populations=max_populations,
        minimum_epsilon=0,
        min_acceptance_rate=0.01,
    )
    elapsed = time.time() - t0

    print(f"  Done in {elapsed/60:.1f} min, final generation {history.max_t}")
    print(f"  Database: {db_path}")

    return history


def plot_results(histories, labels, ground_truth, output_prefix):
    params = list(LOG_PRIORS.keys())
    param_labels = {
        'init_migration_rate': r'$\log_{10}\,\phi$',
        'meth_rate': r'$\log_{10}\,\gamma_m$',
        'demeth_rate': r'$\log_{10}\,\gamma_d$',
        'mu_driver_birth': r'$\log_{10}\,\mu$',
        's_driver_birth': r'$s$',
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    colors = ['steelblue', 'coral', 'seagreen']

    for i, param in enumerate(params):
        ax = axes[i]
        gt_val = ground_truth[param]

        for j, (history, label) in enumerate(zip(histories, labels)):
            df, w = history.get_distribution(m=0, t=history.max_t)
            vals = df[param].values
            weights = np.asarray(w)
            weights = weights / weights.sum()

            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(vals, weights=weights)
                x_grid = np.linspace(
                    LOG_PRIORS[param][0],
                    LOG_PRIORS[param][1],
                    300,
                )
                density = kde(x_grid)
                ax.fill_between(x_grid, density, alpha=0.3, color=colors[j])
                ax.plot(x_grid, density, color=colors[j], lw=1.5, label=label)
            except Exception:
                ax.hist(vals, bins=30, weights=weights, density=True,
                        alpha=0.3, color=colors[j], label=label)

        ax.axvline(gt_val, color='red', ls='--', lw=2, label='Ground truth')

        ax.axvspan(*LOG_PRIORS[param], alpha=0.03, color='grey')

        ax.set_xlabel(param_labels[param], fontsize=12)
        if i == 0:
            ax.set_ylabel('Posterior density')
        ax.legend(fontsize=8)

        if param in ['mu_driver_birth', 's_driver_birth']:
            ax.set_title(f'{param_labels[param]} (target: identifiable?)',
                         fontsize=10, color='darkred')
        else:
            ax.set_title(param_labels[param], fontsize=10)

    axes[5].set_visible(False)

    plt.suptitle(
        'Synthetic parameter recovery: standard vs extended summary statistics\n'
        f'Ground truth: s={GROUND_TRUTH["s_driver_birth"]}, '
        f'$\\mu$={GROUND_TRUTH["mu_driver_birth"]}',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / f'{output_prefix}.pdf',
                bbox_inches='tight', dpi=300)
    fig.savefig(FIGURE_DIR / f'{output_prefix}.png',
                bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved {output_prefix}.pdf")

    # convergence across generations
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        gt_val = ground_truth[param]

        for j, (history, label) in enumerate(zip(histories, labels)):
            medians = []
            ci_lo = []
            ci_hi = []
            gens = list(range(history.max_t + 1))

            for t in gens:
                df, w = history.get_distribution(m=0, t=t)
                vals = df[param].values
                weights = np.asarray(w)
                weights = weights / weights.sum()

                sorted_idx = np.argsort(vals)
                sorted_vals = vals[sorted_idx]
                cum_w = np.cumsum(weights[sorted_idx])

                def wq(p):
                    idx = np.searchsorted(cum_w, p)
                    return sorted_vals[min(idx, len(sorted_vals)-1)]

                medians.append(wq(0.5))
                ci_lo.append(wq(0.05))
                ci_hi.append(wq(0.95))

            ax.fill_between(gens, ci_lo, ci_hi, alpha=0.2, color=colors[j])
            ax.plot(gens, medians, 'o-', color=colors[j], markersize=3,
                    label=label)

        ax.axhline(gt_val, color='red', ls='--', lw=1.5)
        ax.set_xlabel('Generation')
        ax.set_ylabel(param_labels[param])

        if i == 0:
            ax.legend(fontsize=8)

    axes[5].set_visible(False)
    plt.suptitle('Posterior convergence across ABC-SMC generations',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / f'{output_prefix}_convergence.pdf',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved {output_prefix}_convergence.pdf")


def print_summary(histories, labels):
    """Print and save posterior medians and 90% intervals for each experiment."""
    db_dir = OUTPUT_DIR / "selection_recovery"

    for history, label in zip(histories, labels):
        print(f"\n{label}")
        df, w = history.get_distribution(m=0, t=history.max_t)
        weights = np.asarray(w)
        weights = weights / weights.sum()

        for param in LOG_PRIORS:
            vals = df[param].values
            sorted_idx = np.argsort(vals)
            sorted_vals = vals[sorted_idx]
            cum_w = np.cumsum(weights[sorted_idx])

            def wq(p, cum_w=cum_w, sorted_vals=sorted_vals):
                idx = np.searchsorted(cum_w, p)
                return sorted_vals[min(idx, len(sorted_vals)-1)]

            gt = GROUND_TRUTH_LOG[param]
            med = wq(0.5)
            lo = wq(0.05)
            hi = wq(0.95)
            prior_range = LOG_PRIORS[param][1] - LOG_PRIORS[param][0]
            ci_width = hi - lo
            narrowing = 1 - (ci_width / prior_range)

            in_ci = lo <= gt <= hi

            print(f"  {param:25s}: median={med:.3f}  "
                  f"90%CI=[{lo:.3f}, {hi:.3f}]  "
                  f"truth={gt:.3f}  "
                  f"in_CI={in_ci}  "
                  f"narrowing={narrowing:.0%}")

    summary = {
        'ground_truth': GROUND_TRUTH,
        'ground_truth_log': {k: float(v) for k, v in GROUND_TRUTH_LOG.items()},
        'experiments': [],
    }

    for history, label in zip(histories, labels):
        df, w = history.get_distribution(m=0, t=history.max_t)
        weights = np.asarray(w) / np.asarray(w).sum()

        exp = {'label': label, 'params': {}}
        for param in LOG_PRIORS:
            vals = df[param].values
            sorted_idx = np.argsort(vals)
            sorted_vals = vals[sorted_idx]
            cum_w = np.cumsum(weights[sorted_idx])

            def wq(p, cum_w=cum_w, sorted_vals=sorted_vals):
                idx = np.searchsorted(cum_w, p)
                return sorted_vals[min(idx, len(sorted_vals)-1)]

            exp['params'][param] = {
                'median': float(wq(0.5)),
                'ci90': [float(wq(0.05)), float(wq(0.95))],
                'ground_truth': float(GROUND_TRUTH_LOG[param]),
            }
        summary['experiments'].append(exp)

    with open(db_dir / 'selection_recovery_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {db_dir / 'selection_recovery_summary.json'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Test selection recovery with polarisation statistic')
    parser.add_argument('-n', '--n_cores', type=int, default=16,
                        help='Number of parallel cores (default: 16)')
    parser.add_argument('-p', '--pop_size', type=int, default=500,
                        help='ABC population size (default: 500)')
    parser.add_argument('-g', '--max_gen', type=int, default=12,
                        help='Max ABC-SMC generations (default: 12)')
    parser.add_argument('--standard-only', action='store_true',
                        help='Run only the standard inference (no new stat)')
    parser.add_argument('--extended-only', action='store_true',
                        help='Run only the extended inference (with new stat)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run a single simulation + distance computation '
                             'to verify everything works, then exit')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip inference, just plot from existing .db files')
    args = parser.parse_args()

    db_dir = OUTPUT_DIR / "selection_recovery"
    db_dir.mkdir(exist_ok=True)

    if args.plot_only:
        histories = []
        labels = []
        std_db = db_dir / "standard.db"
        ext_db = db_dir / "extended.db"
        if std_db.exists() and not args.extended_only:
            h = pyabc.History("sqlite:///" + str(std_db))
            histories.append(h)
            labels.append("Standard (L2 + Wass)")
        if ext_db.exists() and not args.standard_only:
            h = pyabc.History("sqlite:///" + str(ext_db))
            histories.append(h)
            labels.append("Extended (+pol. het.)")
        if not histories:
            print("No .db files found in", db_dir)
            sys.exit(1)
        print(f"Plotting from {len(histories)} database(s)...")
        plot_results(histories, labels, GROUND_TRUTH_LOG,
                     'selection_recovery_test')
        print_summary(histories, labels)
        sys.exit(0)

    if not METHDEMON_BIN.exists():
        print(f"methdemon binary not found at {METHDEMON_BIN}")
        sys.exit(1)
    print(f"methdemon binary: {METHDEMON_BIN}")

    print("Simulating synthetic data...")
    print(f"  Ground truth: phi={GROUND_TRUTH['init_migration_rate']}, "
          f"gamma_m={GROUND_TRUTH['meth_rate']}, "
          f"gamma_d={GROUND_TRUTH['demeth_rate']}, "
          f"mu={GROUND_TRUTH['mu_driver_birth']}, "
          f"s={GROUND_TRUTH['s_driver_birth']}")

    observation = None
    attempts = 0
    while observation is None or len(observation) != 8:
        attempts += 1
        if attempts > 10:
            print("Failed to simulate valid synthetic data in 10 attempts")
            sys.exit(1)
        observation = simulate(GROUND_TRUTH)
        if observation is not None:
            print(f"  Attempt {attempts}: got {len(observation)} glands")


    obs_pols = []
    for _, row in observation.iterrows():
        pol = np.mean(np.abs(2 * row['AverageArray'] - 1))
        obs_pols.append(pol)
    print(f"  Observed polarisations: {[f'{p:.4f}' for p in obs_pols]}")
    print(f"  Polarisation variance: {np.var(obs_pols):.6f}")
    print(f"  Polarisation range: {max(obs_pols) - min(obs_pols):.4f}")

    observed_sum_stat = {"data": observation}

    if args.dry_run:
        print("Dry run")
        test_params = GROUND_TRUTH_LOG.copy()
        sim_result = simulate_abc_log(test_params)
        sim_data = sim_result['data']
        if sim_data is None:
            print("  Simulation failed")
            sys.exit(1)
        print(f"  Simulation OK: {len(sim_data)} glands")

        for fn in [l2_distance, overall_wasserstein,
                   polarisation_heterogeneity, polarisation_profile_distance]:
            d = fn(observed_sum_stat, sim_result)
            print(f"  {fn.__name__:35s} = {d:.6f}")

        prior = pyabc.Distribution(
            **{key: pyabc.RV("uniform", a, b - a)
               for key, (a, b) in LOG_PRIORS.items()},
        )
        distance = pyabc.AdaptiveAggregatedDistance(
            [l2_distance, overall_wasserstein,
             polarisation_heterogeneity, polarisation_profile_distance])
        sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=2)
        abc = pyabc.ABCSMC(
            simulate_abc_log, prior, distance,
            population_size=5, sampler=sampler,
        )

        import tempfile as _tf
        _dry_db = Path(_tf.mkdtemp()) / "dry_run.db"
        abc.new(
            db="sqlite:///" + str(_dry_db),
            observed_sum_stat=observed_sum_stat,
            gt_par=GROUND_TRUTH_LOG,
        )
        print("  pyabc ABCSMC.new() OK")

        # one tiny generation to check multiprocessing works
        history = abc.run(max_nr_populations=1, minimum_epsilon=0,
                          min_acceptance_rate=0.001)
        print(f"  pyabc.run() OK, {len(history.get_distribution(m=0, t=0)[0])} particles")
        import shutil
        shutil.rmtree(_dry_db.parent, ignore_errors=True)

        print("Dry run OK")
        sys.exit(0)

    histories = []
    labels = []

    db_dir = OUTPUT_DIR / "selection_recovery"
    db_dir.mkdir(exist_ok=True)

    if not args.extended_only:
        h_standard = run_inference(
            name="STANDARD (L2 + Wasserstein)",
            distance_fns=[l2_distance, overall_wasserstein],
            observed=observation,
            db_path=db_dir / "standard.db",
            n_cores=args.n_cores,
            pop_size=args.pop_size,
            max_populations=args.max_gen,
        )
        histories.append(h_standard)
        labels.append("Standard (L2 + Wass)")

    if not args.standard_only:
        h_extended = run_inference(
            name="EXTENDED (L2 + Wasserstein + polarisation heterogeneity)",
            distance_fns=[l2_distance, overall_wasserstein,
                          polarisation_heterogeneity,
                          polarisation_profile_distance],
            observed=observation,
            db_path=db_dir / "extended.db",
            n_cores=args.n_cores,
            pop_size=args.pop_size,
            max_populations=args.max_gen,
        )
        histories.append(h_extended)
        labels.append("Extended (+pol. het.)")

    if histories:
        plot_results(histories, labels, GROUND_TRUTH_LOG,
                     'selection_recovery_test')

    print_summary(histories, labels)
    print("Done.")


if __name__ == '__main__':
    main()

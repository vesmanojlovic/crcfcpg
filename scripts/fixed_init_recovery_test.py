#!/usr/bin/env python3
"""Ground truth recovery test, random vs fixed initial array for all sims."""

import multiprocessing
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
from scipy.stats import wasserstein_distance, gaussian_kde

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

LOG_PRIORS = {
    'meth_rate': (-4, -2),
    'demeth_rate': (-4, -2),
    'init_migration_rate': (-3.3, -1),
    'mu_driver_birth': (-5, -2),
    's_driver_birth': (0, 0.2),
}

N_FCPG = 1200  # fCpG loci (config value; internal = 2*N_FCPG)


def generate_random_init_array():
    """Random binary initial array as in methdemon, alleles of locus j at j and j + N_FCPG."""
    return np.random.randint(0, 2, size=2 * N_FCPG)


def beta_to_init_array(beta_values):
    """Binary init array with each allele methylated with probability beta."""
    assert len(beta_values) == N_FCPG
    arr = np.zeros(2 * N_FCPG, dtype=int)
    for j in range(N_FCPG):
        b = beta_values[j]
        arr[j] = 1 if np.random.random() < b else 0
        arr[j + N_FCPG] = 1 if np.random.random() < b else 0
    return arr


def save_init_array(arr, path):
    np.savetxt(path, arr, fmt='%d')


def write_config(params, output_dir, init_array_path=None):
    """Write config.dat, using init_array_path as manual_array if given."""
    with open(CONFIG_TEMPLATE, 'r') as f:
        lines = f.readlines()

    seed = np.random.randint(1, 2**15 - 1)
    params_with_seed = dict(params)
    params_with_seed['seed'] = seed

    updated = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if parts and parts[0] == 'manual_array' and init_array_path is not None:
            updated.append(f"    manual_array {init_array_path}\n")
        elif parts and parts[0] in params_with_seed:
            val = params_with_seed[parts[0]]
            if isinstance(val, (int, np.integer)):
                updated.append(f"    {parts[0]} {val}\n")
            else:
                updated.append(f"    {parts[0]} {val:.10f}\n")
        else:
            updated.append(line)

    config_path = os.path.join(output_dir, 'config.dat')
    with open(config_path, 'w') as f:
        f.writelines(updated)
    return config_path


def simulate(params, init_array_path=None):
    tmpdir = tempfile.mkdtemp(prefix='fixinit_')
    try:
        config_path = write_config(params, tmpdir, init_array_path)
        result = subprocess.run(
            [str(METHDEMON_BIN), tmpdir, 'config.dat'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        )
        if result.returncode != 0:
            return None

        data_path = os.path.join(tmpdir, 'final_demes.csv')
        if not os.path.exists(data_path):
            return None

        df = pd.read_csv(data_path, header=0)
        df.AverageArray = df.AverageArray.apply(
            lambda x: np.fromstring(x, sep=";"))
        df = df[np.floor(df.Generation) == np.floor(df.Generation.max())]
        df = df.reset_index(drop=True)
        return df

    except Exception:
        return None
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# set by main() before ABC runs
_INIT_ARRAY_PATH = None


def simulate_abc_log_random(log_params):
    params = {k: 10**v for k, v in log_params.items() if k != 's_driver_birth'}
    params['s_driver_birth'] = log_params['s_driver_birth']
    params['deme_carrying_capacity'] = 100
    res = simulate(params, init_array_path=None)
    return {"data": res}


def simulate_abc_log_fixed(log_params):
    params = {k: 10**v for k, v in log_params.items() if k != 's_driver_birth'}
    params['s_driver_birth'] = log_params['s_driver_birth']
    params['deme_carrying_capacity'] = 100
    res = simulate(params, init_array_path=_INIT_ARRAY_PATH)
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
    df1, df2 = dict1['data'], dict2['data']
    if df1 is None or df2 is None:
        return 1000.0
    x = compute_deme_matrix(df1)
    y = compute_deme_matrix(df2)
    return float(np.sqrt(np.sum((x - y) ** 2) / 2))


def overall_wasserstein(dict1, dict2):
    df1, df2 = dict1['data'], dict2['data']
    if df1 is None or df2 is None:
        return 1000.0
    if len(df1) != 8 or len(df2) != 8:
        return 1000.0
    df1 = df1.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
    df2 = df2.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
    res = 0
    for i in range(8):
        res += wasserstein_distance(
            df1.iloc[i].AverageArray, df2.iloc[i].AverageArray)
    return float(res)


def run_inference(name, simulate_fn, distance_fns, observed, db_path,
                  n_cores=16, pop_size=500, max_populations=12):
    print(f"\nRunning {name}...")
    print(f"pop {pop_size}, max gen {max_populations}, cores {n_cores}, "
          f"distances {[f.__name__ for f in distance_fns]}")

    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a)
           for key, (a, b) in LOG_PRIORS.items()})

    distance = pyabc.AdaptiveAggregatedDistance(distance_fns)
    sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_cores)

    abc = pyabc.ABCSMC(
        simulate_fn, prior, distance,
        population_size=pop_size, sampler=sampler)

    abc.new(
        db="sqlite:///" + str(db_path),
        observed_sum_stat={"data": observed},
        gt_par=GROUND_TRUTH_LOG)

    t0 = time.time()
    history = abc.run(
        max_nr_populations=max_populations,
        minimum_epsilon=0, min_acceptance_rate=0.01)
    elapsed = time.time() - t0

    print(f"Finished in {elapsed/60:.1f} min, final generation {history.max_t}")
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
    colors = ['steelblue', 'coral', 'seagreen']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        gt_val = ground_truth[param]

        for j, (history, label) in enumerate(zip(histories, labels)):
            df, w = history.get_distribution(m=0, t=history.max_t)
            vals = df[param].values
            weights = np.asarray(w)
            weights = weights / weights.sum()

            try:
                kde = gaussian_kde(vals, weights=weights)
                x_grid = np.linspace(*LOG_PRIORS[param], 300)
                density = kde(x_grid)
                ax.fill_between(x_grid, density, alpha=0.3, color=colors[j])
                ax.plot(x_grid, density, color=colors[j], lw=1.5, label=label)
            except Exception:
                ax.hist(vals, bins=30, weights=weights, density=True,
                        alpha=0.3, color=colors[j], label=label)

        ax.axvline(gt_val, color='red', ls='--', lw=2, label='Ground truth')
        ax.set_xlabel(param_labels[param], fontsize=12)
        if i == 0:
            ax.set_ylabel('Posterior density')
        ax.legend(fontsize=8)

        if param in ['mu_driver_birth', 's_driver_birth']:
            ax.set_title(f'{param_labels[param]} (target)', fontsize=10,
                         color='darkred')
        else:
            ax.set_title(param_labels[param], fontsize=10)

    axes[5].set_visible(False)
    plt.suptitle(
        'Parameter recovery: random init vs fixed init\n'
        f'Ground truth: s={GROUND_TRUTH["s_driver_birth"]}, '
        f'$\\mu$={GROUND_TRUTH["mu_driver_birth"]}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / f'{output_prefix}.pdf',
                bbox_inches='tight', dpi=300)
    fig.savefig(FIGURE_DIR / f'{output_prefix}.png',
                bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved {output_prefix}.pdf")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        for j, (history, label) in enumerate(zip(histories, labels)):
            medians, ci_lo, ci_hi = [], [], []
            gens = list(range(history.max_t + 1))
            for t in gens:
                df, w = history.get_distribution(m=0, t=t)
                vals = df[param].values
                weights = np.asarray(w)
                weights = weights / weights.sum()
                sorted_idx = np.argsort(vals)
                sorted_vals = vals[sorted_idx]
                cum_w = np.cumsum(weights[sorted_idx])

                def wq(p, c=cum_w, s=sorted_vals):
                    idx = np.searchsorted(c, p)
                    return s[min(idx, len(s)-1)]

                medians.append(wq(0.5))
                ci_lo.append(wq(0.05))
                ci_hi.append(wq(0.95))

            ax.fill_between(gens, ci_lo, ci_hi, alpha=0.2, color=colors[j])
            ax.plot(gens, medians, 'o-', color=colors[j], markersize=3,
                    label=label)

        ax.axhline(ground_truth[param], color='red', ls='--', lw=1.5)
        ax.set_xlabel('Generation')
        ax.set_ylabel(param_labels[param])
        if i == 0:
            ax.legend(fontsize=8)

    axes[5].set_visible(False)
    plt.suptitle('Posterior convergence: random vs fixed init',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / f'{output_prefix}_convergence.pdf',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved {output_prefix}_convergence.pdf")


def print_summary(histories, labels):
    db_dir = OUTPUT_DIR / "fixed_init_recovery"

    for history, label in zip(histories, labels):
        print(f"\n{label}:")
        df, w = history.get_distribution(m=0, t=history.max_t)
        weights = np.asarray(w)
        weights = weights / weights.sum()

        for param in LOG_PRIORS:
            vals = df[param].values
            sorted_idx = np.argsort(vals)
            sorted_vals = vals[sorted_idx]
            cum_w = np.cumsum(weights[sorted_idx])

            def wq(p, c=cum_w, s=sorted_vals):
                idx = np.searchsorted(c, p)
                return s[min(idx, len(s)-1)]

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
                  f"in_CI={'YES' if in_ci else 'NO'}  "
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

            def wq(p, c=cum_w, s=sorted_vals):
                idx = np.searchsorted(c, p)
                return s[min(idx, len(s)-1)]

            exp['params'][param] = {
                'median': float(wq(0.5)),
                'ci90': [float(wq(0.05)), float(wq(0.95))],
                'ground_truth': float(GROUND_TRUTH_LOG[param]),
            }
        summary['experiments'].append(exp)

    with open(db_dir / 'fixed_init_recovery_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {db_dir}")


def main():
    global _INIT_ARRAY_PATH

    import argparse
    parser = argparse.ArgumentParser(
        description='Ground truth recovery: random vs fixed initial array')
    parser.add_argument('-n', '--n_cores', type=int, default=20)
    parser.add_argument('-p', '--pop_size', type=int, default=500)
    parser.add_argument('-g', '--max_gen', type=int, default=12)
    parser.add_argument('--random-only', action='store_true',
                        help='Run only random-init experiment')
    parser.add_argument('--fixed-only', action='store_true',
                        help='Run only fixed-init experiment')
    parser.add_argument('--dry-run', action='store_true',
                        help='Quick sanity check')
    parser.add_argument('--plot-only', action='store_true',
                        help='Plot from existing .db files')
    args = parser.parse_args()

    db_dir = OUTPUT_DIR / "fixed_init_recovery"
    db_dir.mkdir(exist_ok=True)

    if args.plot_only:
        histories, labels = [], []
        for db_name, label in [("random_init.db", "Random init"),
                                ("fixed_init.db", "Fixed init")]:
            db_path = db_dir / db_name
            if db_path.exists():
                if args.fixed_only and 'Random' in label:
                    continue
                if args.random_only and 'Fixed' in label:
                    continue
                h = pyabc.History("sqlite:///" + str(db_path))
                histories.append(h)
                labels.append(label)
        if not histories:
            print("No .db files found")
            sys.exit(1)
        plot_results(histories, labels, GROUND_TRUTH_LOG,
                     'fixed_init_recovery_test')
        print_summary(histories, labels)
        sys.exit(0)

    if not METHDEMON_BIN.exists():
        print(f"methdemon binary not found at {METHDEMON_BIN}")
        sys.exit(1)
    print(f"methdemon binary: {METHDEMON_BIN}")

    init_arr = generate_random_init_array()
    init_arr_path = db_dir / "ground_truth_init_array.txt"
    save_init_array(init_arr, init_arr_path)
    _INIT_ARRAY_PATH = str(init_arr_path)
    print(f"Saved init array ({len(init_arr)} values) to {init_arr_path}")
    init_betas = (init_arr[:N_FCPG] + init_arr[N_FCPG:]) / 2.0
    print(f"Init array mean beta {np.mean(init_betas):.4f}")

    print("Simulating observed data with fixed init...")
    print(f"Ground truth: phi={GROUND_TRUTH['init_migration_rate']}, "
          f"gamma_m={GROUND_TRUTH['meth_rate']}, "
          f"gamma_d={GROUND_TRUTH['demeth_rate']}, "
          f"mu={GROUND_TRUTH['mu_driver_birth']}, "
          f"s={GROUND_TRUTH['s_driver_birth']}")

    observation = None
    attempts = 0
    while observation is None or len(observation) != 8:
        attempts += 1
        if attempts > 10:
            print("Failed to generate valid observed data")
            sys.exit(1)
        observation = simulate(GROUND_TRUTH, init_array_path=_INIT_ARRAY_PATH)
        if observation is not None:
            print(f"Attempt {attempts}: {len(observation)} glands")

    print(f"Observed data has {len(observation)} glands")
    for _, row in observation.iterrows():
        arr = row['AverageArray']
        print(f"  side {row['Side']}: mean beta {np.mean(arr):.4f}")

    observed_sum_stat = {"data": observation}

    if args.dry_run:
        print("\nDry run...")
        test_params = GROUND_TRUTH_LOG.copy()
        res_rand = simulate_abc_log_random(test_params)
        print(f"  Random init sim: {'OK' if res_rand['data'] is not None else 'FAIL'}")

        res_fixed = simulate_abc_log_fixed(test_params)
        print(f"  Fixed init sim:  {'OK' if res_fixed['data'] is not None else 'FAIL'}")

        for fn in [l2_distance, overall_wasserstein]:
            d_r = fn(observed_sum_stat, res_rand)
            d_f = fn(observed_sum_stat, res_fixed)
            print(f"  {fn.__name__:25s}: random={d_r:.4f}  fixed={d_f:.4f}")

        # check pyabc wiring with a tiny run
        prior = pyabc.Distribution(
            **{key: pyabc.RV("uniform", a, b - a)
               for key, (a, b) in LOG_PRIORS.items()})
        distance = pyabc.AdaptiveAggregatedDistance(
            [l2_distance, overall_wasserstein])

        for sim_fn, label in [(simulate_abc_log_random, "random"),
                               (simulate_abc_log_fixed, "fixed")]:
            sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=2)
            abc = pyabc.ABCSMC(sim_fn, prior, distance,
                               population_size=5, sampler=sampler)
            _dry_db = Path(tempfile.mkdtemp()) / f"dry_{label}.db"
            abc.new(db="sqlite:///" + str(_dry_db),
                    observed_sum_stat=observed_sum_stat,
                    gt_par=GROUND_TRUTH_LOG)
            history = abc.run(max_nr_populations=1, minimum_epsilon=0,
                              min_acceptance_rate=0.001)
            n_particles = len(history.get_distribution(m=0, t=0)[0])
            print(f"  pyabc run ({label}): OK, {n_particles} particles")
            import shutil
            shutil.rmtree(_dry_db.parent, ignore_errors=True)

        print("Dry run passed")
        sys.exit(0)

    histories, labels = [], []

    if not args.fixed_only:
        h = run_inference(
            name="RANDOM INIT (standard)",
            simulate_fn=simulate_abc_log_random,
            distance_fns=[l2_distance, overall_wasserstein],
            observed=observation,
            db_path=db_dir / "random_init.db",
            n_cores=args.n_cores,
            pop_size=args.pop_size,
            max_populations=args.max_gen)
        histories.append(h)
        labels.append("Random init")

    if not args.random_only:
        h = run_inference(
            name="FIXED INIT (same array for all sims)",
            simulate_fn=simulate_abc_log_fixed,
            distance_fns=[l2_distance, overall_wasserstein],
            observed=observation,
            db_path=db_dir / "fixed_init.db",
            n_cores=args.n_cores,
            pop_size=args.pop_size,
            max_populations=args.max_gen)
        histories.append(h)
        labels.append("Fixed init")

    if histories:
        plot_results(histories, labels, GROUND_TRUTH_LOG,
                     'fixed_init_recovery_test')
    print_summary(histories, labels)
    print("Done")


if __name__ == '__main__':
    main()

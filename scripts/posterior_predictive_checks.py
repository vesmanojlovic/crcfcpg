#!/usr/bin/env python3
"""Posterior predictive checks of the inter-gland distances for each tumour."""

import argparse
import os
import subprocess
import tempfile
import sqlite3
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from methabc.utils import repo_dir
warnings.filterwarnings('ignore')

REPO_DIR = Path(__file__).resolve().parents[1]
METHABC_DIR = Path(repo_dir)
METHDEMON_BIN = METHABC_DIR / "resources/methdemon/bin/methdemon"
CONFIG_TEMPLATE = METHABC_DIR / "resources/config_template.dat"
TUMOUR_DATA_DIR = REPO_DIR / "data" / "individual"
OUTPUT_DIR = REPO_DIR / "outputs"
FIGURE_DIR = REPO_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# same as extract_posteriors.py, minus H
DB_MAP = {
    'D': ('2025_reruns_D.db', True),
    'E': ('log_model_E.db', True),
    'F': ('2025_reruns_F.db', True),
    'I': ('log_model_I.db', True),
    'J': ('log_model_J.db', True),
    'M': ('2025_reruns_M.db', True),
    'P': ('2025_reruns_P.db', True),
    'S': ('log_model_S.db', True),
    'U': ('2025_reruns_U.db', True),
    'X': ('log_model_X.db', True),
}

PARAM_NAMES_DB = ['init_migration_rate', 'meth_rate', 'demeth_rate',
                  'mu_driver_birth', 's_driver_birth']


def extract_final_posterior(db_path, is_log_scale):
    """Final-generation posterior particles and weights from a pyabc db."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(id) FROM abc_smc")
    abc_smc_id = cursor.fetchone()[0]

    cursor.execute(
        "SELECT id, t FROM populations WHERE abc_smc_id=? ORDER BY t DESC LIMIT 1",
        (abc_smc_id,)
    )
    row = cursor.fetchone()
    if row is None:
        cursor.execute("SELECT id, t FROM populations ORDER BY t DESC LIMIT 1")
        row = cursor.fetchone()
    final_pop_id, final_t = row

    cursor.execute("SELECT id FROM models WHERE population_id=?", (final_pop_id,))
    model_id = cursor.fetchone()[0]

    cursor.execute("SELECT id, w FROM particles WHERE model_id=?", (model_id,))
    particles = cursor.fetchall()

    records = []
    for pid, w in particles:
        cursor.execute("SELECT name, value FROM parameters WHERE particle_id=?", (pid,))
        params = {r[0]: r[1] for r in cursor.fetchall()}
        params['weight'] = w
        records.append(params)

    conn.close()
    df = pd.DataFrame(records)

    # simulator wants natural scale
    if is_log_scale:
        for col in ['init_migration_rate', 'meth_rate', 'demeth_rate', 'mu_driver_birth']:
            if col in df.columns:
                df[f'{col}_log10'] = df[col]
                df[col] = 10 ** df[col]

    return df


def write_config(params_dict, output_dir, seed=None):
    if seed is None:
        seed = np.random.randint(1, 2**15 - 1)

    with open(CONFIG_TEMPLATE, 'r') as f:
        lines = f.readlines()

    param_updates = {
        'init_migration_rate': params_dict['init_migration_rate'],
        'meth_rate': params_dict['meth_rate'],
        'demeth_rate': params_dict['demeth_rate'],
        'mu_driver_birth': params_dict['mu_driver_birth'],
        's_driver_birth': params_dict['s_driver_birth'],
        'mu_driver_migration': 1e-10,  # avoid Poisson(0) assertion
        's_driver_migration': 1e-10,
        'seed': int(seed),
    }

    updated_lines = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if parts and parts[0] in param_updates:
            val = param_updates[parts[0]]
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


def run_methdemon(params_dict, sim_id=0):
    """Run methdemon once and return the distance summaries, or None on failure."""
    tmpdir = tempfile.mkdtemp(prefix=f'ppc_{sim_id}_')

    try:
        write_config(params_dict, tmpdir)
        result = subprocess.run(
            [str(METHDEMON_BIN), tmpdir, 'config.dat'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )

        if result.returncode != 0:
            return None

        output_file = os.path.join(tmpdir, 'final_demes.csv')
        if not os.path.exists(output_file):
            return None

        df = pd.read_csv(output_file)

        # last 8 rows are the final generation
        if len(df) < 8:
            return None

        final_df = df.tail(8).reset_index(drop=True)

        arrays = []
        sides = []
        for _, row in final_df.iterrows():
            arr = np.array([float(x) for x in str(row['AverageArray']).split(';') if x.strip()])
            arrays.append(arr)
            sides.append(row['Side'])

        if len(arrays) != 8:
            return None

        # mean squared difference
        n = len(arrays)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = (arrays[i] - arrays[j]) ** 2
                dist_matrix[i, j] = np.mean(diff)

        within_dists = []
        between_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                if sides[i] == sides[j]:
                    within_dists.append(dist_matrix[i, j])
                else:
                    between_dists.append(dist_matrix[i, j])

        return {
            'dist_matrix': dist_matrix,
            'within_dists': within_dists,
            'between_dists': between_dists,
            'all_pairwise': dist_matrix[np.triu_indices(n, k=1)].tolist(),
            'mean_within': np.mean(within_dists) if within_dists else np.nan,
            'mean_between': np.mean(between_dists) if between_dists else np.nan,
            'mean_all': np.mean(dist_matrix[np.triu_indices(n, k=1)]),
        }

    except Exception as e:
        return None
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_single_ppc(args):
    params_dict, sim_id = args
    return run_methdemon(params_dict, sim_id)


def load_observed_data(tumour_id):
    data_path = TUMOUR_DATA_DIR / f'tumour_{tumour_id}.csv'
    if not data_path.exists():
        return None

    data = pd.read_csv(data_path, index_col=0).dropna()

    # side A glands first, then side B
    columns_a = sorted([col for col in data.columns if 'A' in col], key=len)
    columns_b = sorted([col for col in data.columns if 'B' in col], key=len)
    other_columns = [col for col in data.columns if 'A' not in col and 'B' not in col]
    sorted_columns = columns_a + columns_b + other_columns
    data = data[sorted_columns]

    n_glands = len(data.columns)
    n_a = len(columns_a)

    dist_matrix = np.zeros((n_glands, n_glands))
    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            diff = (data[col1] - data[col2]) ** 2
            dist_matrix[i, j] = diff.mean()

    within_dists = []
    between_dists = []
    for i in range(n_glands):
        for j in range(i + 1, n_glands):
            same_side = (i < n_a and j < n_a) or (i >= n_a and j >= n_a)
            if same_side:
                within_dists.append(dist_matrix[i, j])
            else:
                between_dists.append(dist_matrix[i, j])

    return {
        'dist_matrix': dist_matrix,
        'within_dists': within_dists,
        'between_dists': between_dists,
        'all_pairwise': dist_matrix[np.triu_indices(n_glands, k=1)].tolist(),
        'mean_within': np.mean(within_dists),
        'mean_between': np.mean(between_dists),
        'mean_all': np.mean(dist_matrix[np.triu_indices(n_glands, k=1)]),
        'n_glands': n_glands,
        'n_a': n_a,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_dir', type=str, required=True,
                        help='Directory containing the inference databases')
    parser.add_argument('--n_sims', type=int, default=100,
                        help='Number of PPC simulations per tumour')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()

    N_SIMS = args.n_sims
    N_WORKERS = args.n_workers

    print(f"Posterior predictive checks: {N_SIMS} simulations per tumour, "
          f"{N_WORKERS} workers")
    print(f"Methdemon binary: {METHDEMON_BIN}")

    all_results = {}

    for tumour_id in sorted(DB_MAP.keys()):
        db_file, is_log = DB_MAP[tumour_id]
        db_path = Path(args.db_dir) / db_file

        if not db_path.exists():
            print(f"{db_file} not found, skipping tumour {tumour_id}")
            continue

        print(f"Tumour {tumour_id}")

        obs = load_observed_data(tumour_id)
        if obs is None:
            print(f"  No observed data for tumour {tumour_id}, skipping")
            continue

        print(f"  Observed: {obs['n_glands']} glands, "
              f"mean within={obs['mean_within']:.4f}, "
              f"mean between={obs['mean_between']:.4f}")

        posterior_df = extract_final_posterior(db_path, is_log)
        weights = posterior_df['weight'].values
        weights = weights / weights.sum()

        indices = np.random.choice(len(posterior_df), size=N_SIMS, p=weights)

        param_sets = []
        for idx in indices:
            row = posterior_df.iloc[idx]
            params_dict = {
                'init_migration_rate': row['init_migration_rate'],
                'meth_rate': row['meth_rate'],
                'demeth_rate': row['demeth_rate'],
                'mu_driver_birth': row['mu_driver_birth'],
                's_driver_birth': row['s_driver_birth'],
            }
            param_sets.append(params_dict)

        print(f"  Running {N_SIMS} simulations...")
        sim_results = []
        tasks = [(ps, i) for i, ps in enumerate(param_sets)]

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(run_single_ppc, task): i
                       for i, task in enumerate(tasks)}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 20 == 0:
                    print(f"    {completed}/{N_SIMS} complete")
                result = future.result()
                if result is not None:
                    sim_results.append(result)

        n_success = len(sim_results)
        print(f"  {n_success}/{N_SIMS} simulations successful")

        if n_success < 10:
            print(f"  Too few successful simulations for tumour {tumour_id}, skipping")
            continue

        sim_mean_within = [r['mean_within'] for r in sim_results]
        sim_mean_between = [r['mean_between'] for r in sim_results]
        sim_mean_all = [r['mean_all'] for r in sim_results]
        sim_all_pairwise = [r['all_pairwise'] for r in sim_results]

        # (n_sims, 28) for 8 glands
        all_sim_pairwise = np.array([r['all_pairwise'] for r in sim_results])

        sim_flat = all_sim_pairwise.flatten()
        obs_flat = np.array(obs['all_pairwise'])

        # coverage: fraction of observed pairwise distances inside the
        # 5-95th percentile of the pooled simulated distances
        p5 = np.percentile(sim_flat, 5)
        p95 = np.percentile(sim_flat, 95)
        coverage = np.mean((obs_flat >= p5) & (obs_flat <= p95))

        sim_within_flat = np.array([d for r in sim_results for d in r['within_dists']])
        sim_between_flat = np.array([d for r in sim_results for d in r['between_dists']])
        obs_within = np.array(obs['within_dists'])
        obs_between = np.array(obs['between_dists'])

        p5w, p95w = np.percentile(sim_within_flat, [5, 95])
        p5b, p95b = np.percentile(sim_between_flat, [5, 95])
        coverage_within = np.mean((obs_within >= p5w) & (obs_within <= p95w))
        coverage_between = np.mean((obs_between >= p5b) & (obs_between <= p95b))

        obs_mean = obs['mean_all']
        sim_mean_median = np.median(sim_mean_all)
        sim_mean_p5 = np.percentile(sim_mean_all, 5)
        sim_mean_p95 = np.percentile(sim_mean_all, 95)
        obs_in_interval = sim_mean_p5 <= obs_mean <= sim_mean_p95

        result = {
            'tumour_id': tumour_id,
            'n_sims': n_success,
            'obs_mean_all': float(obs_mean),
            'obs_mean_within': float(obs['mean_within']),
            'obs_mean_between': float(obs['mean_between']),
            'sim_mean_all_median': float(sim_mean_median),
            'sim_mean_all_90ci': [float(sim_mean_p5), float(sim_mean_p95)],
            'sim_mean_within_median': float(np.median(sim_mean_within)),
            'sim_mean_between_median': float(np.median(sim_mean_between)),
            'coverage_all': float(coverage),
            'coverage_within': float(coverage_within),
            'coverage_between': float(coverage_between),
            'obs_in_90_interval': bool(obs_in_interval),
        }

        all_results[tumour_id] = result

        print(f"  Coverage (all pairwise): {coverage:.2%}")
        print(f"  Coverage (within-side): {coverage_within:.2%}")
        print(f"  Coverage (between-side): {coverage_between:.2%}")
        print(f"  Observed mean dist: {obs_mean:.4f}")
        print(f"  Simulated mean dist: {sim_mean_median:.4f} "
              f"[{sim_mean_p5:.4f}, {sim_mean_p95:.4f}]")
        print(f"  Observed in 90% CI: {obs_in_interval}")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        ax = axes[0]
        ax.hist(sim_flat, bins=50, density=True, alpha=0.5,
                color='steelblue', label='Simulated')
        for od in obs_flat:
            ax.axvline(od, color='red', alpha=0.3, lw=0.8)
        ax.axvline(obs_flat[0], color='red', alpha=0.3, lw=0.8,
                   label='Observed')
        ax.set_xlabel('Pairwise distance')
        ax.set_ylabel('Density')
        ax.set_title(f'Tumour {tumour_id}: all pairwise')
        ax.legend(fontsize=8)

        ax = axes[1]
        bp = ax.boxplot([sim_within_flat, sim_between_flat],
                        labels=['Within-side\n(simulated)', 'Between-side\n(simulated)'],
                        patch_artist=True,
                        boxprops=dict(alpha=0.5))
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('coral')
        ax.scatter([1] * len(obs_within), obs_within, c='red', s=30,
                   zorder=5, label='Observed', marker='x')
        ax.scatter([2] * len(obs_between), obs_between, c='red', s=30,
                   zorder=5, marker='x')
        ax.set_ylabel('Pairwise distance')
        ax.set_title(f'Within vs between')
        ax.legend(fontsize=8)

        ax = axes[2]
        ax.hist(sim_mean_all, bins=30, density=True, alpha=0.5,
                color='steelblue', label='Simulated mean')
        ax.axvline(obs_mean, color='red', lw=2, label=f'Observed ({obs_mean:.4f})')
        ax.axvline(sim_mean_p5, color='grey', ls='--', lw=1, label='90% CI')
        ax.axvline(sim_mean_p95, color='grey', ls='--', lw=1)
        ax.set_xlabel('Mean pairwise distance')
        ax.set_ylabel('Density')
        ax.set_title(f'Mean distance PPC')
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(FIGURE_DIR / f'ppc_tumour_{tumour_id}.png', dpi=200,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved ppc_tumour_{tumour_id}.png")

    if all_results:
        tumour_ids = sorted(all_results.keys())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        x = np.arange(len(tumour_ids))
        width = 0.25
        c_all = [all_results[t]['coverage_all'] for t in tumour_ids]
        c_within = [all_results[t]['coverage_within'] for t in tumour_ids]
        c_between = [all_results[t]['coverage_between'] for t in tumour_ids]

        ax.bar(x - width, c_all, width, label='All pairs', color='steelblue', alpha=0.8)
        ax.bar(x, c_within, width, label='Within-side', color='coral', alpha=0.8)
        ax.bar(x + width, c_between, width, label='Between-side', color='seagreen', alpha=0.8)
        ax.axhline(0.9, color='grey', ls='--', lw=1, alpha=0.5, label='90% target')
        ax.set_xticks(x)
        ax.set_xticklabels(tumour_ids)
        ax.set_ylabel('Coverage (fraction in 90% CI)')
        ax.set_xlabel('Tumour')
        ax.set_title('A', loc='left', fontweight='bold', fontsize=14)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

        ax = axes[1]
        obs_means = [all_results[t]['obs_mean_all'] for t in tumour_ids]
        sim_medians = [all_results[t]['sim_mean_all_median'] for t in tumour_ids]
        sim_lo = [all_results[t]['sim_mean_all_90ci'][0] for t in tumour_ids]
        sim_hi = [all_results[t]['sim_mean_all_90ci'][1] for t in tumour_ids]
        yerr = [np.array(sim_medians) - np.array(sim_lo),
                np.array(sim_hi) - np.array(sim_medians)]

        ax.errorbar(x, sim_medians, yerr=yerr, fmt='o', color='steelblue',
                    capsize=4, label='Simulated (median + 90% CI)')
        ax.scatter(x, obs_means, c='red', s=60, zorder=5, marker='x',
                   label='Observed')
        ax.set_xticks(x)
        ax.set_xticklabels(tumour_ids)
        ax.set_ylabel('Mean pairwise distance')
        ax.set_xlabel('Tumour')
        ax.set_title('B', loc='left', fontweight='bold', fontsize=14)
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(FIGURE_DIR / 'fig_ppc_summary.pdf', bbox_inches='tight')
        fig.savefig(FIGURE_DIR / 'fig_ppc_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved fig_ppc_summary.pdf")

    with open(OUTPUT_DIR / 'ppc_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {OUTPUT_DIR / 'ppc_results.json'}")

    print()
    print(f"{'Tumour':<8} {'Cov(all)':<10} {'Cov(within)':<12} {'Cov(between)':<13} "
          f"{'Obs mean':<10} {'Sim median':<11} {'In 90%CI?':<10}")
    for tid in sorted(all_results.keys()):
        r = all_results[tid]
        print(f"{tid:<8} {r['coverage_all']:<10.2%} {r['coverage_within']:<12.2%} "
              f"{r['coverage_between']:<13.2%} {r['obs_mean_all']:<10.4f} "
              f"{r['sim_mean_all_median']:<11.4f} {'Yes' if r['obs_in_90_interval'] else 'No':<10}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Supplementary figures: distance heatmaps for all tumours and synthetic parameter recovery."""

import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "data" / "individual"
FIGURE_DIR = REPO_DIR / "figures"
OUTPUT_DIR = REPO_DIR / "outputs"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_DB = None  # set with --ground_truth_db

TUMOUR_IDS = ["D", "E", "F", "I", "J", "M", "P", "S", "U", "X"]

# ground truth from methabc/scripts/abc-example.py
GROUND_TRUTH = {
    'fission_rate': np.log10(0.009),      # -2.0458
    'meth_rate': np.log10(0.0022),        # -2.6576
    'demeth_rate': np.log10(0.0018),      # -2.7447
    'driver_mutation_rate': np.log10(0.0001),  # -4.0
    'selective_advantage': 0.1,            # not log-transformed
}


def assign_side(col_name, tumour_id):
    name_upper = col_name.upper().strip()
    second_char = name_upper[len(tumour_id)]
    return "A" if second_char == "A" else "B"


HEATMAP_VMAX = 0.15


def generate_heatmaps():
    """Generate centred 3-4-3 grid of distance matrix heatmaps."""
    rows_layout = [
        ["D", "E", "F"],
        ["I", "J", "M", "U"],
        ["P", "S", "X"],
    ]
    # 8-column grid: 4 items span cols 0-1,2-3,4-5,6-7;
    # 3 items span cols 1-2,3-4,5-6 (centred)
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 8, figure=fig, hspace=0.15, wspace=1.0)

    col_spans = {
        3: [(1, 3), (3, 5), (5, 7)],
        4: [(0, 2), (2, 4), (4, 6), (6, 8)],
    }

    all_axes = []
    im = None
    for r, row_tids in enumerate(rows_layout):
        spans = col_spans[len(row_tids)]
        for c, tid in enumerate(row_tids):
            c0, c1 = spans[c]
            ax = fig.add_subplot(gs[r, c0:c1])
            all_axes.append(ax)

            fpath = DATA_DIR / f"tumour_{tid}.csv"
            df = pd.read_csv(fpath, index_col=0).dropna()

            glands = df.columns.tolist()
            sides = {g: assign_side(g, tid) for g in glands}
            sorted_glands = sorted(glands, key=lambda g: (sides[g], g))
            df = df[sorted_glands]

            n = len(sorted_glands)
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    msd = np.nanmean(
                        (df.iloc[:, i].values - df.iloc[:, j].values) ** 2)
                    dist[i, j] = msd
                    dist[j, i] = msd

            im = ax.imshow(dist, cmap='magma', vmin=0, vmax=HEATMAP_VMAX)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(sorted_glands, rotation=45, ha='right',
                               fontsize=7)
            ax.set_yticklabels(sorted_glands, fontsize=7)
            ax.tick_params(axis='y', pad=1, length=2)
            ax.set_title(f'Tumour {tid}', fontsize=9, fontweight='bold')

            n_a = sum(1 for s in sides.values() if s == 'A')
            if 0 < n_a < n:
                ax.axhline(n_a - 0.5, color='white', linewidth=1.5)
                ax.axvline(n_a - 0.5, color='white', linewidth=1.5)

    # dashed line between carcinomas (rows 1-2) and adenomas (row 3)
    row2_bottoms = [a.get_position().y0 for a in all_axes[3:7]]
    row3_tops = [a.get_position().y1 for a in all_axes[7:]]
    sep_y = (min(row2_bottoms) + max(row3_tops)) / 2
    fig.add_artist(plt.Line2D([0.02, 0.75], [sep_y, sep_y],
                              transform=fig.transFigure, color='grey',
                              ls='--', lw=1.0, clip_on=False))
    fig.text(0.01, sep_y + 0.01, 'Carcinomas', fontsize=7, color='grey',
             transform=fig.transFigure, va='bottom')
    fig.text(0.01, sep_y - 0.01, 'Adenomas', fontsize=7, color='grey',
             transform=fig.transFigure, va='top')

    fig.colorbar(im, ax=all_axes, shrink=0.4,
                 label=r'Mean-squared $\beta$ difference', pad=0.03)
    fig.savefig(FIGURE_DIR / 'sfig_all_heatmaps.pdf', bbox_inches='tight',
                dpi=300)
    fig.savefig(FIGURE_DIR / 'sfig_all_heatmaps.png', bbox_inches='tight',
                dpi=200)
    plt.close(fig)
    print("Saved sfig_all_heatmaps.pdf")


def extract_all_generations(db_path):
    """Extract posterior from every generation for convergence plot."""
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("SELECT MAX(id) FROM abc_smc")
    abc_id = c.fetchone()[0]

    c.execute("SELECT id, t FROM populations WHERE abc_smc_id=? AND t>=0 ORDER BY t",
              (abc_id,))
    populations = c.fetchall()

    gen_data = {}
    param_names_map = {
        'init_migration_rate': 'fission_rate',
        'meth_rate': 'meth_rate',
        'demeth_rate': 'demeth_rate',
        'mu_driver_birth': 'driver_mutation_rate',
        's_driver_birth': 'selective_advantage',
    }

    for pop_id, t in populations:
        c.execute("SELECT id FROM models WHERE population_id=?", (pop_id,))
        model_id = c.fetchone()[0]

        c.execute("SELECT id, w FROM particles WHERE model_id=?", (model_id,))
        particles = c.fetchall()

        records = []
        for pid, w in particles:
            c.execute("SELECT name, value FROM parameters WHERE particle_id=?", (pid,))
            params = {r[0]: r[1] for r in c.fetchall()}
            params['weight'] = w
            records.append(params)

        df = pd.DataFrame(records)
        rename = {k: v for k, v in param_names_map.items() if k in df.columns}
        df = df.rename(columns=rename)
        gen_data[t] = df

    conn.close()
    return gen_data


def generate_recovery_figure():
    if GROUND_TRUTH_DB is None or not GROUND_TRUTH_DB.exists():
        print(f"{GROUND_TRUTH_DB} not found, skipping recovery figure")
        return

    gen_data = extract_all_generations(GROUND_TRUTH_DB)
    final_gen = max(gen_data.keys())
    final_df = gen_data[final_gen]

    # identifiable parameters
    params_show = ['fission_rate', 'meth_rate', 'demeth_rate']
    param_labels = [
        r'$\log_{10}$ fission rate $\phi$',
        r'$\log_{10}$ methylation rate $\gamma_m$',
        r'$\log_{10}$ demethylation rate $\gamma_d$',
    ]

    params_all = ['fission_rate', 'meth_rate', 'demeth_rate',
                  'driver_mutation_rate', 'selective_advantage']
    param_labels_all = [
        r'$\log_{10}\,\phi$',
        r'$\log_{10}\,\gamma_m$',
        r'$\log_{10}\,\gamma_d$',
        r'$\log_{10}\,\mu$',
        r'$s$',
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].text(-0.15, 1.1, 'A', transform=axes[0, 0].transAxes,
                    fontsize=16, fontweight='bold', va='bottom')
    axes[1, 0].text(-0.15, 1.1, 'B', transform=axes[1, 0].transAxes,
                    fontsize=16, fontweight='bold', va='bottom')

    # top row: final posteriors, bottom row: convergence over generations
    for i, (param, label) in enumerate(zip(params_show, param_labels)):
        ax = axes[0, i]
        vals = final_df[param].values
        weights = final_df['weight'].values
        weights = weights / weights.sum()

        kde = stats.gaussian_kde(vals, weights=weights)
        x_grid = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 300)
        density = kde(x_grid)

        ax.fill_between(x_grid, density, alpha=0.5, color='steelblue')
        ax.plot(x_grid, density, color='steelblue', lw=1.5)

        gt = GROUND_TRUTH[param]
        ax.axvline(gt, color='red', lw=2, ls='--', label=f'Truth: {gt:.3f}')

        sorted_idx = np.argsort(vals)
        cum_w = np.cumsum(weights[sorted_idx])
        median_idx = np.searchsorted(cum_w, 0.5)
        median_val = vals[sorted_idx[min(median_idx, len(vals)-1)]]
        ax.axvline(median_val, color='navy', lw=1.5, ls=':',
                   label=f'Median: {median_val:.3f}')

        ax.set_xlabel(label)
        ax.set_ylabel('Posterior density')
        ax.legend(fontsize=8)
        ax.set_title(f'Generation {final_gen}', fontsize=10)

    for i, (param, label) in enumerate(zip(params_show, param_labels)):
        ax = axes[1, i]
        generations = sorted(gen_data.keys())

        medians = []
        ci_lo = []
        ci_hi = []

        for t in generations:
            df = gen_data[t]
            vals = df[param].values
            w = df['weight'].values
            w = w / w.sum()

            sorted_idx = np.argsort(vals)
            sorted_vals = vals[sorted_idx]
            cum_w = np.cumsum(w[sorted_idx])

            def wq(p):
                idx = np.searchsorted(cum_w, p)
                return sorted_vals[min(idx, len(sorted_vals)-1)]

            medians.append(wq(0.5))
            ci_lo.append(wq(0.05))
            ci_hi.append(wq(0.95))

        ax.fill_between(generations, ci_lo, ci_hi, alpha=0.3, color='steelblue')
        ax.plot(generations, medians, 'o-', color='steelblue', markersize=4)
        ax.axhline(GROUND_TRUTH[param], color='red', ls='--', lw=1.5)
        ax.set_xlabel('ABC-SMC generation')
        ax.set_ylabel(label)
        ax.set_title('Posterior convergence', fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'sfig_recovery.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(FIGURE_DIR / 'sfig_recovery.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("Saved sfig_recovery.pdf")

    # all params, identifiable ones on the top row
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5), constrained_layout=True)
    for i, (param, label) in enumerate(zip(params_all, param_labels_all)):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        vals = final_df[param].values
        weights = final_df['weight'].values
        weights = weights / weights.sum()

        kde = stats.gaussian_kde(vals, weights=weights)
        x_grid = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)
        density = kde(x_grid)

        ax.fill_between(x_grid, density, alpha=0.5, color='steelblue')
        ax.plot(x_grid, density, color='steelblue', lw=1.5)
        ax.axvline(GROUND_TRUTH[param], color='red', lw=2, ls='--',
                   label='Ground truth')
        ax.set_xlabel(label, fontsize=10)
        if col == 0:
            ax.set_ylabel('Posterior density', fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=9)

        if param in ['driver_mutation_rate', 'selective_advantage']:
            ax.set_title('Non-identifiable', fontsize=9, color='grey')
        else:
            ax.set_title('Identifiable', fontsize=9, color='darkgreen')

    axes[1, 2].set_visible(False)
    fig.savefig(FIGURE_DIR / 'sfig_recovery_all_params.pdf',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved sfig_recovery_all_params.pdf")


def generate_recovery_corner():
    """Corner plot: marginal posteriors on diagonal, 2D joint densities below."""
    if GROUND_TRUTH_DB is None or not GROUND_TRUTH_DB.exists():
        print(f"{GROUND_TRUTH_DB} not found, skipping corner figure")
        return

    gen_data = extract_all_generations(GROUND_TRUTH_DB)
    final_gen = max(gen_data.keys())
    final_df = gen_data[final_gen]

    params = ['fission_rate', 'meth_rate', 'demeth_rate',
              'driver_mutation_rate', 'selective_advantage']
    labels = [r'$\log_{10}\,\phi$', r'$\log_{10}\,\gamma_m$',
              r'$\log_{10}\,\gamma_d$', r'$\log_{10}\,\mu$', r'$s$']
    priors = {
        'fission_rate': (-3.3, -1),
        'meth_rate': (-4, -2),
        'demeth_rate': (-4, -2),
        'driver_mutation_rate': (-5, -2),
        'selective_advantage': (0, 0.2),
    }

    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(7.2, 7.2))

    weights = final_df['weight'].values
    weights = weights / weights.sum()

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                vals = final_df[params[i]].values
                kde = stats.gaussian_kde(vals, weights=weights)
                lo, hi = priors[params[i]]
                x_grid = np.linspace(lo, hi, 300)
                density = kde(x_grid)

                ax.fill_between(x_grid, density, alpha=0.4, color='steelblue')
                ax.plot(x_grid, density, color='steelblue', lw=1.2)
                ax.axvline(GROUND_TRUTH[params[i]], color='red', lw=1.5,
                           ls='--')

                prior_height = 1.0 / (hi - lo)
                ax.axhline(prior_height, color='grey', lw=1, ls=':',
                           alpha=0.7)
                ax.set_xlim(lo, hi)
                ax.set_yticks([])
            else:
                x = final_df[params[j]].values
                y = final_df[params[i]].values
                lo_x, hi_x = priors[params[j]]
                lo_y, hi_y = priors[params[i]]

                h, xedges, yedges = np.histogram2d(
                    x, y, bins=40, range=[[lo_x, hi_x], [lo_y, hi_y]],
                    weights=weights, density=True)
                ax.pcolormesh(xedges, yedges, h.T, cmap='Blues', shading='auto')
                ax.axvline(GROUND_TRUTH[params[j]], color='red', lw=1, ls='--',
                           alpha=0.7)
                ax.axhline(GROUND_TRUTH[params[i]], color='red', lw=1, ls='--',
                           alpha=0.7)
                ax.set_xlim(lo_x, hi_x)
                ax.set_ylim(lo_y, hi_y)

            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=10)
                ax.tick_params(axis='x', labelsize=7, rotation=45)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=10)
            elif j == 0 and i == 0:
                ax.set_ylabel('Density', fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='red', ls='--', lw=1.5, label='Ground truth'),
        Line2D([0], [0], color='steelblue', lw=2, label='Posterior'),
        Line2D([0], [0], color='grey', ls=':', lw=1, label='Prior'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', fontsize=9,
               frameon=True, bbox_to_anchor=(0.88, 0.88))

    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    fig.savefig(FIGURE_DIR / 'sfig_recovery_all_params_alt.pdf',
                bbox_inches='tight', dpi=300)
    fig.savefig(FIGURE_DIR / 'sfig_recovery_all_params_alt.png',
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved sfig_recovery_all_params_alt.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_db', type=str, default=None,
                        help='Database of the synthetic recovery run (recovery figures are skipped without it)')
    args = parser.parse_args()
    if args.ground_truth_db is not None:
        GROUND_TRUTH_DB = Path(args.ground_truth_db)

    print("Generating distance matrix heatmaps...")
    generate_heatmaps()

    print("Generating synthetic recovery figure...")
    generate_recovery_figure()

    print("Generating recovery corner plot...")
    generate_recovery_corner()

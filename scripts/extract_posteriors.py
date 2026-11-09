#!/usr/bin/env python3
"""Final ABC-SMC posteriors for all tumours, cohort statistics and Fig 3."""
# usage: python scripts/extract_posteriors.py --db_dir <dir>

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

REPO_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_DIR / "outputs"
FIGURE_DIR = REPO_DIR / "figures"
PATIENT_CSV = REPO_DIR / "data" / "patient_info.csv"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# tumour_id -> (db_filename, is_log_scale)
# inference_S.db is not on log scale, so log_model_S.db is used instead
DB_MAP = {
    'D': ('2025_reruns_D.db', True),
    'E': ('log_model_E.db', True),
    'F': ('2025_reruns_F.db', True),
    # H excluded, being re-run on log scale
    'I': ('log_model_I.db', True),
    'J': ('log_model_J.db', True),
    'M': ('2025_reruns_M.db', True),
    'P': ('2025_reruns_P.db', True),
    'S': ('log_model_S.db', True),
    'U': ('2025_reruns_U.db', True),
    'X': ('log_model_X.db', True),
}

PARAM_NAMES = {
    'init_migration_rate': 'fission_rate',
    'meth_rate': 'meth_rate',
    'demeth_rate': 'demeth_rate',
    'mu_driver_birth': 'driver_mutation_rate',
    's_driver_birth': 'selective_advantage',
}

K = 100  # carrying capacity

ADENOMAS = {'P', 'S', 'X'}
CARCINOMA_COLOR = 'steelblue'
ADENOMA_COLOR = 'coral'


def load_patient_info():
    df = pd.read_csv(PATIENT_CSV, encoding='latin-1')
    df = df[df['CANCER'] != 'our code'].copy()
    df['size cm'] = pd.to_numeric(df['size cm'], errors='coerce')
    df['stage'] = df['stage'].astype(str).str.strip()

    # codes I, G, Z are duplicated; keep the first row (I=3.6cm stage 3 as in thesis)
    tumour_info = {}
    seen = set()
    for _, row in df.iterrows():
        code = row['CANCER'].strip()
        if code in DB_MAP and code not in seen:
            tumour_info[code] = {
                'age': row['age'],
                'sex': row['sex'],
                'dx': row['dx'],
                'stage': row['stage'],
                'size_cm': row['size cm'],
            }
            seen.add(code)
    return tumour_info


def extract_final_posterior(db_path, is_log_scale):
    """Final population of a pyabc database, with rate parameters on log10 scale."""
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
        # some databases don't have abc_smc_id set properly
        cursor.execute("SELECT id, t FROM populations ORDER BY t DESC LIMIT 1")
        row = cursor.fetchone()
    final_pop_id, final_t = row

    cursor.execute("SELECT id FROM models WHERE population_id=?", (final_pop_id,))
    model_id = cursor.fetchone()[0]

    cursor.execute(
        "SELECT id, w FROM particles WHERE model_id=?",
        (model_id,)
    )
    particles = cursor.fetchall()

    particle_ids = [p[0] for p in particles]
    weights = {p[0]: p[1] for p in particles}

    records = []
    for pid in particle_ids:
        cursor.execute(
            "SELECT name, value FROM parameters WHERE particle_id=?",
            (pid,)
        )
        params = {row[0]: row[1] for row in cursor.fetchall()}
        params['weight'] = weights[pid]
        params['particle_id'] = pid
        records.append(params)

    conn.close()
    df = pd.DataFrame(records)

    rename = {k: v for k, v in PARAM_NAMES.items() if k in df.columns}
    df = df.rename(columns=rename)

    if not is_log_scale:
        for col in ['fission_rate', 'meth_rate', 'demeth_rate', 'driver_mutation_rate']:
            if col in df.columns:
                df[col] = np.log10(df[col])
        # s_driver_birth is not a rate, so stays on natural scale

    return df, final_t


def compute_summary(df, param_name):
    """Weighted median and 50%/90% credible intervals of a log10 parameter."""
    values = df[param_name].values
    weights = df['weight'].values
    weights = weights / weights.sum()

    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cum_weights = np.cumsum(sorted_weights)

    def weighted_percentile(p):
        idx = np.searchsorted(cum_weights, p)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    median = weighted_percentile(0.5)
    ci50_lo = weighted_percentile(0.25)
    ci50_hi = weighted_percentile(0.75)
    ci90_lo = weighted_percentile(0.05)
    ci90_hi = weighted_percentile(0.95)

    return {
        'log10_median': median,
        'log10_ci50': (ci50_lo, ci50_hi),
        'log10_ci90': (ci90_lo, ci90_hi),
        'median': 10**median,
        'ci50': (10**ci50_lo, 10**ci50_hi),
        'ci90': (10**ci90_lo, 10**ci90_hi),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_dir', type=str, required=True,
                        help='Directory containing the inference databases')
    args = parser.parse_args()

    patient_info = load_patient_info()
    print(f"Patient info for {sorted(patient_info.keys())}")
    for tid, info in sorted(patient_info.items()):
        print(f"  {tid}: size={info['size_cm']} cm, stage={info['stage']}, "
              f"age={info['age']}, sex={info['sex']}, dx={info['dx']}")

    all_posteriors = {}
    all_summaries = {}
    missing = []

    for tumour_id in sorted(DB_MAP.keys()):
        db_file, is_log = DB_MAP[tumour_id]
        db_path = Path(args.db_dir) / db_file

        if not db_path.exists():
            print(f"{db_file} not found, skipping tumour {tumour_id}")
            missing.append(tumour_id)
            continue

        print(f"\nTumour {tumour_id} ({db_file}, log_scale={is_log})")
        df, final_t = extract_final_posterior(db_path, is_log)
        all_posteriors[tumour_id] = df
        print(f"  Final generation: t={final_t}, particles: {len(df)}")

        summaries = {}
        for param in ['fission_rate', 'meth_rate', 'demeth_rate']:
            s = compute_summary(df, param)
            summaries[param] = s
            print(f"  {param}:")
            print(f"    log10 median: {s['log10_median']:.4f}")
            print(f"    median (natural): {s['median']:.6f}")
            print(f"    50% CI: [{s['ci50'][0]:.6f}, {s['ci50'][1]:.6f}]")
            print(f"    90% CI: [{s['ci90'][0]:.6f}, {s['ci90'][1]:.6f}]")

        all_summaries[tumour_id] = summaries

    if missing:
        print(f"\nMissing databases: {missing}")

    print("\nComparison with thesis estimates:")
    thesis_fission = {'E': 0.0025, 'I': 0.017, 'J': 0.003444, 'S': 0.007, 'X': 0.0032}
    thesis_meth = {'E': 0.0021, 'I': 0.005, 'J': 0.0012, 'S': 0.001, 'X': 0.0008}
    thesis_demeth = {'E': 0.002, 'I': 0.0009, 'J': 0.0018, 'S': 0.0033, 'X': 0.0025}

    for tid in ['E', 'I', 'J', 'S', 'X']:
        if tid in all_summaries:
            s = all_summaries[tid]
            print(f"\n  Tumour {tid}:")
            print(f"    Fission rate: computed={s['fission_rate']['median']:.6f}, "
                  f"thesis={thesis_fission[tid]}")
            print(f"    Meth rate:    computed={s['meth_rate']['median']:.6f}, "
                  f"thesis={thesis_meth[tid]}")
            print(f"    Demeth rate:  computed={s['demeth_rate']['median']:.6f}, "
                  f"thesis={thesis_demeth[tid]}")

    print("\nSummary table:")

    rows = []
    for tid in sorted(all_summaries.keys()):
        s = all_summaries[tid]
        info = patient_info.get(tid, {})
        rows.append({
            'Tumour': tid,
            'Size (cm)': info.get('size_cm', 'n/a'),
            'Stage': info.get('stage', 'n/a'),
            'Age': info.get('age', 'n/a'),
            'Sex': info.get('sex', 'n/a'),
            'Dx': info.get('dx', 'n/a'),
            'Fission rate (median)': f"{s['fission_rate']['median']:.6f}",
            'Fission rate 50% CI': f"[{s['fission_rate']['ci50'][0]:.6f}, {s['fission_rate']['ci50'][1]:.6f}]",
            'Fission rate 90% CI': f"[{s['fission_rate']['ci90'][0]:.6f}, {s['fission_rate']['ci90'][1]:.6f}]",
            'Meth rate (median)': f"{s['meth_rate']['median']:.6f}",
            'Meth rate 50% CI': f"[{s['meth_rate']['ci50'][0]:.6f}, {s['meth_rate']['ci50'][1]:.6f}]",
            'Demeth rate (median)': f"{s['demeth_rate']['median']:.6f}",
            'Demeth rate 50% CI': f"[{s['demeth_rate']['ci50'][0]:.6f}, {s['demeth_rate']['ci50'][1]:.6f}]",
            'Per-gland fission rate': f"{s['fission_rate']['median'] * K:.4f}",
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / 'posterior_summaries.csv', index=False)
    print(summary_df.to_string(index=False))

    tumour_ids_with_size = [tid for tid in all_summaries
                           if tid in patient_info and not pd.isna(patient_info[tid].get('size_cm'))]

    fission_medians = [all_summaries[tid]['fission_rate']['median'] for tid in tumour_ids_with_size]
    tumour_sizes = [patient_info[tid]['size_cm'] for tid in tumour_ids_with_size]

    print(f"\nTumours with fission rate and size data: {tumour_ids_with_size}")
    print(f"Fission medians: {[f'{v:.6f}' for v in fission_medians]}")
    print(f"Tumour sizes: {tumour_sizes}")

    rho, p_val = stats.spearmanr(fission_medians, tumour_sizes)
    print(f"\nSpearman correlation (fission rate vs tumour size):")
    print(f"  rho = {rho:.4f}, p = {p_val:.4f}")
    print(f"  n = {len(tumour_ids_with_size)}")

    log_fission_medians = [all_summaries[tid]['fission_rate']['log10_median']
                           for tid in tumour_ids_with_size]
    rho_log, p_val_log = stats.spearmanr(log_fission_medians, tumour_sizes)
    print(f"\nSpearman correlation (log10 fission rate vs tumour size):")
    print(f"  rho = {rho_log:.4f}, p = {p_val_log:.4f}")

    meth_medians = [all_summaries[tid]['meth_rate']['median'] for tid in sorted(all_summaries.keys())]
    demeth_medians = [all_summaries[tid]['demeth_rate']['median'] for tid in sorted(all_summaries.keys())]

    cv_meth = np.std(meth_medians) / np.mean(meth_medians)
    cv_demeth = np.std(demeth_medians) / np.mean(demeth_medians)
    cv_fission = np.std(fission_medians) / np.mean(fission_medians)

    print(f"\nCoefficient of variation across tumours:")
    print(f"  Methylation rate:   CV = {cv_meth:.3f}")
    print(f"  Demethylation rate: CV = {cv_demeth:.3f}")
    print(f"  Fission rate:       CV = {cv_fission:.3f}")

    log_meth_medians = [all_summaries[tid]['meth_rate']['log10_median']
                        for tid in sorted(all_summaries.keys())]
    log_demeth_medians = [all_summaries[tid]['demeth_rate']['log10_median']
                          for tid in sorted(all_summaries.keys())]

    sd_log_meth = np.std(log_meth_medians)
    sd_log_demeth = np.std(log_demeth_medians)
    sd_log_fission = np.std(log_fission_medians)
    range_log_meth = max(log_meth_medians) - min(log_meth_medians)
    range_log_demeth = max(log_demeth_medians) - min(log_demeth_medians)
    range_log_fission = max(log_fission_medians) - min(log_fission_medians)

    print(f"\nStandard deviation on log10 scale:")
    print(f"  Methylation rate:   SD = {sd_log_meth:.3f}, range = {range_log_meth:.3f}")
    print(f"  Demethylation rate: SD = {sd_log_demeth:.3f}, range = {range_log_demeth:.3f}")
    print(f"  Fission rate:       SD = {sd_log_fission:.3f}, range = {range_log_fission:.3f}")

    min_fission = min(fission_medians)
    max_fission = max(fission_medians)
    fold_change = max_fission / min_fission
    print(f"\nFission rate range:")
    print(f"  Min: {min_fission:.6f} (tumour {tumour_ids_with_size[fission_medians.index(min_fission)]})")
    print(f"  Max: {max_fission:.6f} (tumour {tumour_ids_with_size[fission_medians.index(max_fission)]})")
    print(f"  Fold change: {fold_change:.1f}x")

    print("\nTumour age estimates:")

    # per-gland fission rate is per-cell rate * K; time to N_final glands from one
    # gland is ln(N_final) / per-gland rate in cell divisions, then converted to
    # real time with a CSC division rate of 1/week to 1/month

    N_final = 1e7  # target number of glands
    csc_rates = {'1/week': 1, '1/2weeks': 0.5, '1/month': 1/4.33}  # divisions per week

    print(f"\nAssuming exponential growth to {N_final:.0e} glands:")
    for tid in sorted(all_summaries.keys()):
        phi = all_summaries[tid]['fission_rate']['median']
        per_gland_phi = phi * K
        divisions_needed = np.log(N_final) / per_gland_phi
        print(f"\n  Tumour {tid}: per-cell phi={phi:.6f}, per-gland phi={per_gland_phi:.4f}")
        print(f"    Cell divisions needed: {divisions_needed:.0f}")
        for rate_name, rate_val in csc_rates.items():
            weeks = divisions_needed / rate_val
            years = weeks / 52
            print(f"    At {rate_name} CSC division: {years:.1f} years ({weeks:.0f} weeks)")

    stats_out = {
        'spearman_fission_size': {
            'rho': float(rho), 'p_value': float(p_val),
            'n': len(tumour_ids_with_size),
        },
        'spearman_log_fission_size': {
            'rho': float(rho_log), 'p_value': float(p_val_log),
        },
        'cv_meth_rate': float(cv_meth),
        'cv_demeth_rate': float(cv_demeth),
        'cv_fission_rate': float(cv_fission),
        'sd_log10_meth': float(sd_log_meth),
        'sd_log10_demeth': float(sd_log_demeth),
        'sd_log10_fission': float(sd_log_fission),
        'fission_fold_change': float(fold_change),
        'fission_min': float(min_fission),
        'fission_max': float(max_fission),
    }

    with open(OUTPUT_DIR / 'cohort_statistics.json', 'w') as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nSaved {OUTPUT_DIR / 'cohort_statistics.json'}")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
    })

    fig, axes = plt.subplots(3, 1, figsize=(7, 12), constrained_layout=True)

    tumour_order = sorted(all_summaries.keys(),
                          key=lambda t: all_summaries[t]['fission_rate']['log10_median'])

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tumour_order)))
    y_positions = np.arange(len(tumour_order))

    for i, tid in enumerate(tumour_order):
        df = all_posteriors[tid]
        vals = df['fission_rate'].values
        weights = df['weight'].values
        weights = weights / weights.sum()

        x_grid = np.linspace(-3.5, -0.8, 500)
        kde = stats.gaussian_kde(vals, weights=weights)
        density = kde(x_grid)
        density = density / density.max() * 0.8

        ax.fill_between(x_grid, i, i + density, alpha=0.7, color=colors[i])
        ax.plot(x_grid, i + density, color='black', lw=0.5)

        median_val = all_summaries[tid]['fission_rate']['log10_median']
        ax.axvline(median_val, ymin=(i) / len(tumour_order),
                   ymax=(i + 0.5) / len(tumour_order),
                   color='red', lw=0.8, ls='--', alpha=0.7)

    ax.set_yticks(y_positions + 0.3)
    ax.set_yticklabels(tumour_order)
    ax.set_xlabel(r'Fission rate (cell$^{-1}$ cell div$^{-1}$)')
    ax.set_ylabel('Tumour')
    ax.set_title('A', loc='left', fontweight='bold', fontsize=14)
    ax.set_xlim(-3.5, -0.8)
    ax.set_xticks([-3, -2, -1])
    ax.set_xticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])

    # prior range
    ax.axvspan(-3.3, -1.0, alpha=0.05, color='grey')
    ax.axvline(-3.3, color='grey', ls=':', lw=0.5, alpha=0.5)
    ax.axvline(-1.0, color='grey', ls=':', lw=0.5, alpha=0.5)

    ax = axes[1]
    tumour_order_meth = sorted(all_summaries.keys(),
                               key=lambda t: all_summaries[t]['meth_rate']['log10_median'])

    for i, tid in enumerate(tumour_order_meth):
        df = all_posteriors[tid]
        vals = df['meth_rate'].values
        weights = df['weight'].values
        weights = weights / weights.sum()

        x_grid = np.linspace(-4.2, -1.8, 500)
        kde = stats.gaussian_kde(vals, weights=weights)
        density = kde(x_grid)
        density = density / density.max() * 0.8

        ax.fill_between(x_grid, i, i + density, alpha=0.7, color=colors[i])
        ax.plot(x_grid, i + density, color='black', lw=0.5)

    ax.set_yticks(y_positions + 0.3)
    ax.set_yticklabels(tumour_order_meth)
    ax.set_xlabel(r'Methylation rate (cell div$^{-1}$)')
    ax.set_ylabel('Tumour')
    ax.set_title('B', loc='left', fontweight='bold', fontsize=14)
    ax.set_xlim(-4.2, -1.8)
    ax.set_xticks([-4, -3, -2])
    ax.set_xticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'])
    ax.axvspan(-4.0, -2.0, alpha=0.05, color='grey')
    ax.axvline(-4.0, color='grey', ls=':', lw=0.5, alpha=0.5)
    ax.axvline(-2.0, color='grey', ls=':', lw=0.5, alpha=0.5)

    ax = axes[2]
    tumour_order_demeth = sorted(all_summaries.keys(),
                                 key=lambda t: all_summaries[t]['demeth_rate']['log10_median'])

    for i, tid in enumerate(tumour_order_demeth):
        df = all_posteriors[tid]
        vals = df['demeth_rate'].values
        weights = df['weight'].values
        weights = weights / weights.sum()

        x_grid = np.linspace(-4.2, -1.8, 500)
        kde = stats.gaussian_kde(vals, weights=weights)
        density = kde(x_grid)
        density = density / density.max() * 0.8

        ax.fill_between(x_grid, i, i + density, alpha=0.7, color=colors[i])
        ax.plot(x_grid, i + density, color='black', lw=0.5)

    ax.set_yticks(y_positions + 0.3)
    ax.set_yticklabels(tumour_order_demeth)
    ax.set_xlabel(r'Demethylation rate (cell div$^{-1}$)')
    ax.set_ylabel('Tumour')
    ax.set_title('C', loc='left', fontweight='bold', fontsize=14)
    ax.set_xlim(-4.2, -1.8)
    ax.set_xticks([-4, -3, -2])
    ax.set_xticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'])
    ax.axvspan(-4.0, -2.0, alpha=0.05, color='grey')
    ax.axvline(-4.0, color='grey', ls=':', lw=0.5, alpha=0.5)
    ax.axvline(-2.0, color='grey', ls=':', lw=0.5, alpha=0.5)

    fig.savefig(FIGURE_DIR / 'fig3a_posterior_ridgeline.pdf', bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'fig3a_posterior_ridgeline.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'fig3a_posterior_ridgeline.pdf'}")

    # side by side version, sized for textwidth
    fig_alt, axes_alt = plt.subplots(1, 3, figsize=(7.2, 5),
                                     constrained_layout=True)

    # same tumour order (by fission rate) in all panels
    alt_order = sorted(all_summaries.keys(),
                       key=lambda t: all_summaries[t]['fission_rate']['log10_median'])
    alt_colors = {tid: ADENOMA_COLOR if tid in ADENOMAS else CARCINOMA_COLOR
                  for tid in alt_order}

    panels = [
        ('fission_rate', r'Fission rate',
         (-3.5, -0.8), [-3, -2, -1],
         [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'], -3.3, -1.0),
        ('meth_rate', r'Methylation rate',
         (-4.2, -1.8), [-4, -3, -2],
         [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'], -4.0, -2.0),
        ('demeth_rate', r'Demethylation rate',
         (-4.2, -1.8), [-4, -3, -2],
         [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'], -4.0, -2.0),
    ]
    n_t = len(alt_order)
    y_pos = np.arange(n_t)

    for pi, (param, plabel, xr, xt, xtl, plo, phi_) in enumerate(panels):
        ax = axes_alt[pi]
        for i, tid in enumerate(alt_order):
            df = all_posteriors[tid]
            vals = df[param].values
            wt = df['weight'].values
            wt = wt / wt.sum()
            x_grid = np.linspace(*xr, 500)
            kde = stats.gaussian_kde(vals, weights=wt)
            density = kde(x_grid)
            density = density / density.max() * 0.8
            ax.fill_between(x_grid, i, i + density, alpha=0.7,
                            color=alt_colors[tid])
            ax.plot(x_grid, i + density, color='black', lw=0.5)

            median_val = all_summaries[tid][param]['log10_median']
            ax.vlines(median_val, i, i + 0.4,
                      color='red', lw=0.8, ls='--', alpha=0.7)

        ax.set_yticks(y_pos + 0.3)
        ax.set_yticklabels(alt_order, fontsize=10)
        ax.set_xlabel(plabel, fontsize=10)
        if pi == 0:
            ax.set_ylabel('Tumour', fontsize=10)
        ax.set_title(['A', 'B', 'C'][pi], loc='left', fontweight='bold',
                     fontsize=12)
        ax.set_xlim(*xr)
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl, fontsize=10)
        ax.tick_params(labelsize=10)
        ax.axvspan(plo, phi_, alpha=0.05, color='grey')
        ax.axvline(plo, color='grey', ls=':', lw=0.5, alpha=0.5)
        ax.axvline(phi_, color='grey', ls=':', lw=0.5, alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CARCINOMA_COLOR, alpha=0.7,
                             label='Carcinoma'),
                       Patch(facecolor=ADENOMA_COLOR, alpha=0.7,
                             label='Adenoma')]
    fig_alt.legend(handles=legend_elements, fontsize=9, ncol=2,
                   loc='lower center', bbox_to_anchor=(0.5, -0.06),
                   frameon=False)

    fig_alt.savefig(FIGURE_DIR / 'fig3a_posterior_ridgeline_alt.pdf',
                    bbox_inches='tight')
    fig_alt.savefig(FIGURE_DIR / 'fig3a_posterior_ridgeline_alt.png',
                    bbox_inches='tight', dpi=300)
    plt.close(fig_alt)
    print(f"Saved "
          f"{FIGURE_DIR / 'fig3a_posterior_ridgeline_alt.pdf'}")

    fig, (ax_scatter, ax_box) = plt.subplots(1, 2, figsize=(7.2, 3.5),
                                             constrained_layout=True)

    sizes_arr = np.array(tumour_sizes)
    fission_arr = np.array(fission_medians)
    labels = tumour_ids_with_size

    scatter_colors = [ADENOMA_COLOR if tid in ADENOMAS else CARCINOMA_COLOR
                      for tid in labels]
    ax_scatter.scatter(sizes_arr, fission_arr, s=40, c=scatter_colors,
                       edgecolors='black', lw=0.5, zorder=5)
    for i, tid in enumerate(labels):
        ax_scatter.annotate(tid, (sizes_arr[i], fission_arr[i]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=9, fontweight='bold')
    ax_scatter.set_xlabel('Tumour size (cm)', fontsize=10)
    ax_scatter.set_ylabel(r'Median fission rate', fontsize=10)
    ax_scatter.set_title('A', loc='left', fontweight='bold', fontsize=12)
    ax_scatter.set_yscale('log')
    ax_scatter.tick_params(labelsize=10)
    ax_scatter.text(0.95, 0.05,
                    f'$\\rho$ = {rho:.2f}, p = {p_val:.2f}',
                    transform=ax_scatter.transAxes, ha='right', va='bottom',
                    fontsize=9)
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CARCINOMA_COLOR,
               markeredgecolor='black', markersize=7, label='Carcinoma'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ADENOMA_COLOR,
               markeredgecolor='black', markersize=7, label='Adenoma'),
    ]
    ax_scatter.legend(handles=legend_els, fontsize=8, loc='upper left',
                      frameon=True, framealpha=0.9)

    labels_epi = sorted(all_summaries.keys())
    n_tumours_epi = len(labels_epi)
    positions_meth = np.arange(n_tumours_epi) * 2.5
    positions_demeth = positions_meth + 0.8
    meth_boxes = [all_posteriors[tid]['meth_rate'].values for tid in labels_epi]
    demeth_boxes = [all_posteriors[tid]['demeth_rate'].values
                    for tid in labels_epi]

    bp_m = ax_box.boxplot(meth_boxes, positions=positions_meth, widths=0.6,
                          patch_artist=True, showfliers=False,
                          boxprops=dict(facecolor='coral', alpha=0.7),
                          medianprops=dict(color='darkred', lw=1.5),
                          whiskerprops=dict(color='coral'),
                          capprops=dict(color='coral'))
    bp_d = ax_box.boxplot(demeth_boxes, positions=positions_demeth, widths=0.6,
                          patch_artist=True, showfliers=False,
                          boxprops=dict(facecolor='steelblue', alpha=0.7),
                          medianprops=dict(color='darkblue', lw=1.5),
                          whiskerprops=dict(color='steelblue'),
                          capprops=dict(color='steelblue'))
    ax_box.set_xticks(positions_meth + 0.4)
    ax_box.set_xticklabels(labels_epi, fontsize=10)
    ax_box.set_ylabel(r'Rate (cell div$^{-1}$)', fontsize=10)
    ax_box.set_xlabel('Tumour', fontsize=10)
    ax_box.set_yticks([-4, -3.5, -3, -2.5, -2])
    ax_box.set_yticklabels([r'$10^{-4}$', r'$10^{-3.5}$', r'$10^{-3}$',
                            r'$10^{-2.5}$', r'$10^{-2}$'], fontsize=10)
    ax_box.legend([bp_m['boxes'][0], bp_d['boxes'][0]],
                  ['Methylation', 'Demethylation'],
                  loc='upper right', fontsize=8, frameon=True)
    ax_box.set_title('B', loc='left', fontweight='bold', fontsize=12)
    ax_box.tick_params(labelsize=10)

    fig.savefig(FIGURE_DIR / 'fig3bc_combined.pdf', bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'fig3bc_combined.png', bbox_inches='tight',
                dpi=300)
    plt.close(fig)
    print(f"Saved "
          f"{FIGURE_DIR / 'fig3bc_combined.pdf'}")

    fig_epi, ax_epi = plt.subplots(figsize=(7.2, 3.6))
    all_tids = sorted(all_summaries.keys())
    carcinoma_tids = sorted(t for t in all_tids if t not in ADENOMAS)
    adenoma_tids = sorted(t for t in all_tids if t in ADENOMAS)
    labels_epi2 = carcinoma_tids + adenoma_tids
    n_carc = len(carcinoma_tids)

    n_t = len(labels_epi2)
    pos_m = np.arange(n_t) * 2.5
    pos_d = pos_m + 0.8
    meth_b = [all_posteriors[tid]['meth_rate'].values for tid in labels_epi2]
    demeth_b = [all_posteriors[tid]['demeth_rate'].values for tid in labels_epi2]

    bp_m2 = ax_epi.boxplot(meth_b, positions=pos_m, widths=0.6,
                            patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor='coral', alpha=0.7),
                            medianprops=dict(color='black', lw=1.5),
                            whiskerprops=dict(color='coral'),
                            capprops=dict(color='coral'))
    bp_d2 = ax_epi.boxplot(demeth_b, positions=pos_d, widths=0.6,
                            patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor='steelblue', alpha=0.7),
                            medianprops=dict(color='black', lw=1.5),
                            whiskerprops=dict(color='steelblue'),
                            capprops=dict(color='steelblue'))

    sep_x = (pos_m[n_carc - 1] + 0.8 + pos_m[n_carc]) / 2
    ax_epi.axvline(sep_x, color='grey', ls='--', lw=1.0)
    ax_epi.text(sep_x - 0.3, ax_epi.get_ylim()[1], 'Carcinomas',
                fontsize=8, color='grey', ha='right', va='top')
    ax_epi.text(sep_x + 0.3, ax_epi.get_ylim()[1], 'Adenomas',
                fontsize=8, color='grey', ha='left', va='top')

    ax_epi.set_xticks(pos_m + 0.4)
    ax_epi.set_xticklabels(labels_epi2, fontsize=10)
    ax_epi.set_ylabel(r'Rate (cell div$^{-1}$)', fontsize=10)
    ax_epi.set_xlabel('Tumour', fontsize=10)
    ax_epi.set_yticks([-4, -3.5, -3, -2.5, -2])
    ax_epi.set_yticklabels([r'$10^{-4}$', r'$10^{-3.5}$', r'$10^{-3}$',
                            r'$10^{-2.5}$', r'$10^{-2}$'], fontsize=10)
    fig_epi.legend([bp_m2['boxes'][0], bp_d2['boxes'][0]],
                   ['Methylation', 'Demethylation'],
                   loc='lower center', ncol=2, fontsize=10,
                   bbox_to_anchor=(0.5, -0.06), frameon=False)
    ax_epi.tick_params(labelsize=10)
    plt.tight_layout()
    fig_epi.savefig(FIGURE_DIR / 'fig3c_epimutation_rates.pdf',
                    bbox_inches='tight')
    fig_epi.savefig(FIGURE_DIR / 'fig3c_epimutation_rates.png',
                    bbox_inches='tight', dpi=300)
    plt.close(fig_epi)
    print(f"Saved "
          f"{FIGURE_DIR / 'fig3c_epimutation_rates.pdf'}")

    for tid in all_posteriors:
        df = all_posteriors[tid][['fission_rate', 'meth_rate', 'demeth_rate',
                                  'driver_mutation_rate', 'selective_advantage', 'weight']]
        df.to_csv(OUTPUT_DIR / f'posterior_{tid}.csv', index=False)
    print(f"Saved {OUTPUT_DIR}/posterior_*.csv")


if __name__ == '__main__':
    main()

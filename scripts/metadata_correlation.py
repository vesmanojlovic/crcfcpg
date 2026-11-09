#!/usr/bin/env python3
"""Correlations between inferred parameters and tumour metadata."""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

REPO_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_DIR / "outputs"
FIGURE_DIR = REPO_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
POSTERIOR_DIR = OUTPUT_DIR


def load_data():
    """Posterior medians and tumour metadata in one DataFrame."""
    summary_df = pd.read_csv(OUTPUT_DIR / 'posterior_summaries.csv')

    df = pd.DataFrame()
    df['tumour'] = summary_df['Tumour']
    df['size_cm'] = summary_df['Size (cm)']
    df['stage'] = summary_df['Stage'].replace('NaN', np.nan).replace('nan', np.nan)
    df['age'] = summary_df['Age']
    df['sex'] = summary_df['Sex']
    df['dx'] = summary_df['Dx'].str.strip()

    df['fission_rate'] = summary_df['Fission rate (median)'].astype(float)
    df['meth_rate'] = summary_df['Meth rate (median)'].astype(float)
    df['demeth_rate'] = summary_df['Demeth rate (median)'].astype(float)
    df['per_gland_fission'] = summary_df['Per-gland fission rate'].astype(float)

    df['log_fission'] = np.log10(df['fission_rate'])
    df['log_meth'] = np.log10(df['meth_rate'])
    df['log_demeth'] = np.log10(df['demeth_rate'])

    df['is_carcinoma'] = (df['dx'].str.lower() == 'cancer').astype(int)
    df['is_female'] = (df['sex'] == 'F').astype(int)

    # stage is missing for adenomas
    df['stage_num'] = pd.to_numeric(df['stage'], errors='coerce')

    return df


def spearman_tests(df):
    print("\nSpearman rank correlations")

    features = ['size_cm', 'age']
    targets = ['log_fission', 'log_meth', 'log_demeth']
    target_labels = ['Fission rate', 'Methylation rate', 'Demethylation rate']

    results = []
    for feat in features:
        for target, label in zip(targets, target_labels):
            mask = ~(df[feat].isna() | df[target].isna())
            rho, p = stats.spearmanr(df.loc[mask, feat], df.loc[mask, target])
            n = mask.sum()
            results.append({
                'feature': feat, 'target': label,
                'rho': rho, 'p_value': p, 'n': n,
            })
            print(f"  {feat} vs {label}: rho={rho:.3f}, p={p:.4f}, n={n}")

    # stage only for carcinomas
    carcinoma_mask = df['is_carcinoma'] == 1
    carcinoma_df = df[carcinoma_mask]
    print(f"Carcinomas only, n={carcinoma_mask.sum()}:")
    for target, label in zip(targets, target_labels):
        mask = ~carcinoma_df['stage_num'].isna()
        if mask.sum() >= 4:
            rho, p = stats.spearmanr(carcinoma_df.loc[mask, 'stage_num'],
                                     carcinoma_df.loc[mask, target])
            results.append({
                'feature': 'stage', 'target': label,
                'rho': rho, 'p_value': p, 'n': mask.sum(),
            })
            print(f"  stage vs {label}: rho={rho:.3f}, p={p:.4f}, n={mask.sum()}")

    return results


def group_tests(df):
    """Mann-Whitney U tests for sex and adenoma vs carcinoma."""
    print("\nGroup comparisons")

    targets = ['log_fission', 'log_meth', 'log_demeth']
    target_labels = ['Fission rate', 'Methylation rate', 'Demethylation rate']

    results = []

    print("Sex (Mann-Whitney U):")
    for target, label in zip(targets, target_labels):
        male = df[df['is_female'] == 0][target].values
        female = df[df['is_female'] == 1][target].values
        if len(male) >= 2 and len(female) >= 2:
            U, p = stats.mannwhitneyu(male, female, alternative='two-sided')
            results.append({
                'feature': 'sex', 'target': label,
                'test': 'Mann-Whitney U', 'statistic': U, 'p_value': p,
                'n_male': len(male), 'n_female': len(female),
                'median_male': np.median(male), 'median_female': np.median(female),
            })
            print(f"    {label}: U={U:.1f}, p={p:.4f} "
                  f"(M median={np.median(male):.3f}, F median={np.median(female):.3f})")

    print("Adenoma vs carcinoma (Mann-Whitney U):")
    for target, label in zip(targets, target_labels):
        adenoma = df[df['is_carcinoma'] == 0][target].values
        carcinoma = df[df['is_carcinoma'] == 1][target].values
        if len(adenoma) >= 2 and len(carcinoma) >= 2:
            U, p = stats.mannwhitneyu(adenoma, carcinoma, alternative='two-sided')
            results.append({
                'feature': 'dx_type', 'target': label,
                'test': 'Mann-Whitney U', 'statistic': U, 'p_value': p,
                'n_adenoma': len(adenoma), 'n_carcinoma': len(carcinoma),
                'median_adenoma': np.median(adenoma),
                'median_carcinoma': np.median(carcinoma),
            })
            print(f"    {label}: U={U:.1f}, p={p:.4f} "
                  f"(adenoma median={np.median(adenoma):.3f}, "
                  f"carcinoma median={np.median(carcinoma):.3f})")

    return results


def xgboost_analysis(df):
    """Gradient boosting feature importances, with LOOCV R^2 and MAE."""
    print("\nGradient boosting feature importance (LOOCV)")

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError:
        print("scikit-learn not available, installing...")
        import subprocess
        subprocess.run(['pip', 'install', 'scikit-learn'], capture_output=True)
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import r2_score, mean_absolute_error

    # stage left out, missing for all adenomas
    feature_cols = ['size_cm', 'age', 'is_female', 'is_carcinoma']
    feature_labels = ['Tumour size', 'Age', 'Sex (female)', 'Carcinoma']

    targets = ['log_fission', 'log_meth', 'log_demeth']
    target_labels = ['Fission rate', 'Methylation rate', 'Demethylation rate']

    results = {}

    for target, label in zip(targets, target_labels):
        X = df[feature_cols].values
        y = df[target].values

        loo = LeaveOneOut()
        y_pred = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = GradientBoostingRegressor(
                n_estimators=50, max_depth=2, learning_rate=0.1,
                random_state=42, subsample=0.8,
            )
            model.fit(X_train, y_train)
            y_pred[test_idx] = model.predict(X_test)

        # importances from a fit to all tumours
        full_model = GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            random_state=42, subsample=0.8,
        )
        full_model.fit(X, y)

        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        importances = full_model.feature_importances_

        results[target] = {
            'r2_loocv': float(r2),
            'mae_loocv': float(mae),
            'feature_importances': {fl: float(imp)
                                    for fl, imp in zip(feature_labels, importances)},
        }

        print(f"{label}: LOOCV R^2 = {r2:.3f}, MAE = {mae:.3f}")
        for fl, imp in sorted(zip(feature_labels, importances),
                              key=lambda x: -x[1]):
            print(f"  {fl}: {imp:.3f}")

    return results


def generate_figures(df):
    """Combined boxplot and scatter figure, then the importance figure."""
    sns.set_style("whitegrid")

    features = ['size_cm', 'age', 'stage_num']
    feature_labels = ['Tumour size (cm)', 'Age at diagnosis', 'Clinical stage']
    targets = ['log_fission', 'log_meth', 'log_demeth']
    target_labels = [r'$\log_{10}$ fission rate',
                     r'$\log_{10}$ meth rate',
                     r'$\log_{10}$ demeth rate']

    n_carc = len(df[df['is_carcinoma'] == 1])
    n_aden = len(df[df['is_carcinoma'] == 0])

    fig, axes = plt.subplots(4, 3, figsize=(7.2, 10.2),
                             gridspec_kw={'height_ratios': [0.9, 1, 1, 1]})

    for i, (target, label) in enumerate(zip(targets, target_labels)):
        ax = axes[0, i]
        adenoma = df[df['is_carcinoma'] == 0][target]
        carcinoma = df[df['is_carcinoma'] == 1][target]

        bp = ax.boxplot([carcinoma, adenoma], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', lw=1.5))
        bp['boxes'][0].set(facecolor='steelblue', alpha=0.6)
        bp['boxes'][1].set(facecolor='coral', alpha=0.6)

        for j, (group, color) in enumerate([(carcinoma, 'steelblue'),
                                             (adenoma, 'coral')]):
            x = np.random.normal(j + 1, 0.04, size=len(group))
            ax.scatter(x, group, c=color, s=25, edgecolors='black',
                       lw=0.5, zorder=5)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['', ''])
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(labelsize=10)

        U, p = stats.mannwhitneyu(carcinoma, adenoma, alternative='two-sided')
        ax.set_title(f'p={p:.3f}', fontsize=10)

    for i, (feat, flabel) in enumerate(zip(features, feature_labels)):
        for j, (target, tlabel) in enumerate(zip(targets, target_labels)):
            ax = axes[j + 1, i]
            mask = ~(df[feat].isna() | df[target].isna())
            plot_df = df[mask]

            colors = ['coral' if c == 0 else 'steelblue'
                      for c in plot_df['is_carcinoma']]

            ax.scatter(plot_df[feat], plot_df[target], c=colors, s=40,
                       edgecolors='black', lw=0.5, zorder=5)

            for _, row in plot_df.iterrows():
                ax.annotate(row['tumour'],
                            (row[feat], row[target]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=8)

            if mask.sum() >= 4:
                rho, p = stats.spearmanr(plot_df[feat], plot_df[target])
                ax.set_title(f'ρ={rho:.2f}, p={p:.3f}', fontsize=10)

            if j == 2:
                ax.set_xlabel(flabel, fontsize=10)
            if i == 0:
                ax.set_ylabel(tlabel, fontsize=10)
            ax.tick_params(labelsize=10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markeredgecolor='black', markeredgewidth=0.5,
               markersize=8, label=f'Carcinoma (n={n_carc})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='coral',
               markeredgecolor='black', markeredgewidth=0.5,
               markersize=8, label=f'Adenoma (n={n_aden})'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, 1.02), frameon=False)

    axes[0, 0].text(-0.15, 1.15, 'A', transform=axes[0, 0].transAxes,
                    fontsize=14, fontweight='bold', va='bottom')
    axes[1, 0].text(-0.15, 1.15, 'B', transform=axes[1, 0].transAxes,
                    fontsize=14, fontweight='bold', va='bottom')

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'fig_metadata_correlations.pdf',
                bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'fig_metadata_correlations.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig_metadata_correlations.pdf")

    try:
        xgb_results = xgboost_analysis(df)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        targets_xgb = ['log_fission', 'log_meth', 'log_demeth']
        target_labels_xgb = ['Fission rate', 'Methylation rate', 'Demethylation rate']

        for i, (target, label) in enumerate(zip(targets_xgb, target_labels_xgb)):
            ax = axes[i]
            res = xgb_results[target]
            importances = res['feature_importances']
            names = list(importances.keys())
            values = list(importances.values())

            sorted_idx = np.argsort(values)
            ax.barh([names[j] for j in sorted_idx],
                    [values[j] for j in sorted_idx],
                    color='steelblue', alpha=0.8)
            ax.set_xlabel('Feature importance')
            ax.set_title(f'{label}\n(LOOCV R²={res["r2_loocv"]:.2f})')

        plt.tight_layout()
        fig.savefig(FIGURE_DIR / 'fig_xgb_importance.pdf',
                    bbox_inches='tight')
        fig.savefig(FIGURE_DIR / 'fig_xgb_importance.png', dpi=200,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"Saved fig_xgb_importance.pdf")

        return xgb_results
    except Exception as e:
        print(f"Gradient boosting analysis failed: {e}")
        return None


def generate_scatter_figure(df):
    from matplotlib.lines import Line2D
    sns.set_style("whitegrid")

    features = ['size_cm', 'age', 'stage_num']
    feature_labels = ['Tumour size (cm)', 'Age at diagnosis', 'Clinical stage']
    targets = ['log_fission', 'log_meth', 'log_demeth']
    target_labels = [r'$\log_{10}$ fission rate',
                     r'$\log_{10}$ meth rate',
                     r'$\log_{10}$ demeth rate']

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.2))
    for i, (feat, flabel) in enumerate(zip(features, feature_labels)):
        for j, (target, tlabel) in enumerate(zip(targets, target_labels)):
            ax = axes[j, i]
            mask = ~(df[feat].isna() | df[target].isna())
            plot_df = df[mask]

            colors = ['coral' if c == 0 else 'steelblue'
                      for c in plot_df['is_carcinoma']]

            ax.scatter(plot_df[feat], plot_df[target], c=colors, s=40,
                       edgecolors='black', lw=0.5, zorder=5)

            for _, row in plot_df.iterrows():
                ax.annotate(row['tumour'],
                            (row[feat], row[target]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=8)

            if mask.sum() >= 4:
                rho, p = stats.spearmanr(plot_df[feat], plot_df[target])
                ax.set_title(f'ρ={rho:.2f}, p={p:.3f}', fontsize=10)

            if j == 2:
                ax.set_xlabel(flabel, fontsize=10)
            if i == 0:
                ax.set_ylabel(tlabel, fontsize=10)
            ax.tick_params(labelsize=10)

    n_carc = len(df[df['is_carcinoma'] == 1])
    n_aden = len(df[df['is_carcinoma'] == 0])
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label=f'Carcinoma (n={n_carc})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='coral',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label=f'Adenoma (n={n_aden})'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.04), frameon=False)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'fig_metadata_scatterplots.pdf',
                bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'fig_metadata_scatterplots.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig_metadata_scatterplots.pdf")


def generate_adenoma_vs_carcinoma(df):
    from matplotlib.lines import Line2D

    targets = ['log_fission', 'log_meth', 'log_demeth']
    target_labels = [r'$\log_{10}$ fission rate',
                     r'$\log_{10}$ meth rate',
                     r'$\log_{10}$ demeth rate']

    adenoma = df[df['is_carcinoma'] == 0]
    carcinoma = df[df['is_carcinoma'] == 1]
    n_carc, n_aden = len(carcinoma), len(adenoma)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3))
    for i, (target, label) in enumerate(zip(targets, target_labels)):
        ax = axes[i]
        vals_c = carcinoma[target].values
        vals_a = adenoma[target].values

        bp = ax.boxplot([vals_c, vals_a], widths=0.5, patch_artist=True,
                        medianprops=dict(color='black', lw=1.5),
                        showfliers=False)
        bp['boxes'][0].set(facecolor='steelblue', alpha=0.6)
        bp['boxes'][1].set(facecolor='coral', alpha=0.6)

        x_c = np.random.normal(1, 0.04, size=len(vals_c))
        x_a = np.random.normal(2, 0.04, size=len(vals_a))
        ax.scatter(x_c, vals_c, c='steelblue', s=25, edgecolors='black',
                   lw=0.5, zorder=5)
        ax.scatter(x_a, vals_a, c='coral', s=25, edgecolors='black',
                   lw=0.5, zorder=5)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['', ''])
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(labelsize=10)

        U, p = stats.mannwhitneyu(vals_c, vals_a, alternative='two-sided')
        ax.set_title(f'p={p:.3f}', fontsize=10)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='steelblue',
               markersize=8, label=f'Carcinoma (n={n_carc})'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='coral',
               markersize=8, label=f'Adenoma (n={n_aden})'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.06), frameon=False)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'fig_adenoma_vs_carcinoma.pdf',
                bbox_inches='tight')
    fig.savefig(FIGURE_DIR / 'fig_adenoma_vs_carcinoma.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig_adenoma_vs_carcinoma.pdf")


def main():
    df = load_data()
    print("Data loaded:")
    print(df[['tumour', 'size_cm', 'stage', 'age', 'sex', 'dx',
              'fission_rate', 'meth_rate', 'demeth_rate']].to_string(index=False))

    spearman_results = spearman_tests(df)
    group_results = group_tests(df)
    xgb_results = generate_figures(df)
    generate_scatter_figure(df)
    generate_adenoma_vs_carcinoma(df)

    all_results = {
        'spearman': spearman_results,
        'group_tests': group_results,
        'xgboost': xgb_results,
    }

    with open(OUTPUT_DIR / 'metadata_correlation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved {OUTPUT_DIR / 'metadata_correlation_results.json'}")

    print("\nSpearman correlations (*** p < 0.05):")
    for r in spearman_results:
        sig = " ***" if r['p_value'] < 0.05 else ""
        print(f"  {r['feature']} vs {r['target']}: "
              f"rho={r['rho']:.3f}, p={r['p_value']:.4f}{sig}")

    print("\nGroup comparisons (*** p < 0.05):")
    for r in group_results:
        sig = " ***" if r['p_value'] < 0.05 else ""
        print(f"  {r['feature']} on {r['target']}: "
              f"p={r['p_value']:.4f}{sig}")

    if xgb_results:
        print("\nGradient boosting LOOCV R^2:")
        for target, label in [('log_fission', 'Fission'),
                               ('log_meth', 'Methylation'),
                               ('log_demeth', 'Demethylation')]:
            r2 = xgb_results[target]['r2_loocv']
            print(f"  {label}: {r2:.3f}")

    any_sig = any(r['p_value'] < 0.05 for r in spearman_results + group_results)
    print(f"\nAny significant associations (p < 0.05): {any_sig}")
    if not any_sig:
        print("n=11, so power to detect moderate effects is low")


if __name__ == '__main__':
    main()

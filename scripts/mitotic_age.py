#!/usr/bin/env python3
"""
Mitotic age of each tumour, from the fission rate posterior and tumour size,
and from within-gland fCpG polarisation.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_DIR = Path(__file__).resolve().parents[1]
POSTERIOR_DIR = REPO_DIR / "outputs"
DATA_DIR_PRIMARY = REPO_DIR / "data/individual"
DATA_DIR_FALLBACK = REPO_DIR / "data/individual"
PATIENT_CSV = REPO_DIR / "data/patient_info.csv"
FIGURE_DIR = REPO_DIR / "figures"
OUTPUT_DIR = REPO_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TUMOUR_IDS = ["D", "E", "F", "I", "J", "M", "P", "S", "U", "X"]
K = 100  # carrying capacity

ADENOMAS = {'P', 'S', 'X'}
CARCINOMA_COLOR = 'steelblue'
ADENOMA_COLOR = 'coral'

# gland geometry, used to estimate N from tumour size
GLAND_DIAMETER_UM = 70
GLAND_DEPTH_UM = 450
GLAND_VOLUME_MM3 = (np.pi * (GLAND_DIAMETER_UM / 2e3)**2
                    * (GLAND_DEPTH_UM / 1e3))  # mm^3


def estimate_n_glands(diameter_cm):
    """Number of glands in a spherical tumour of the given diameter."""
    radius_mm = diameter_cm * 10 / 2
    volume_mm3 = (4 / 3) * np.pi * radius_mm**3
    return volume_mm3 / GLAND_VOLUME_MM3


def load_tumour_sizes():
    pi = pd.read_csv(PATIENT_CSV, encoding="latin-1")
    sizes = {}
    for tid in TUMOUR_IDS:
        row = pi[pi["CANCER"] == tid]
        if len(row) > 0:
            sizes[tid] = float(row.iloc[0]["size cm"])
    return sizes


def load_posterior(tumour_id):
    """Posterior for a tumour, parameters on log10 scale."""
    path = POSTERIOR_DIR / f"posterior_{tumour_id}.csv"
    return pd.read_csv(path)


def weighted_quantile(values, weights, q):
    idx = np.argsort(values)
    sorted_vals = values[idx]
    cum_w = np.cumsum(weights[idx])
    cum_w /= cum_w[-1]
    return sorted_vals[np.searchsorted(cum_w, q)]


def load_gland_data(tumour_id):
    """fCpG beta values for a tumour (loci x glands)."""
    path = DATA_DIR_PRIMARY / f"tumour_{tumour_id}.csv"
    if not path.exists():
        path = DATA_DIR_FALLBACK / f"tumour_{tumour_id}.csv"
    df = pd.read_csv(path, index_col=0).dropna()
    return df


def compute_mitotic_age_posteriors(sizes):
    """Posterior of mitotic age per tumour, t_div = ln(N) / (phi * K)."""
    results = {}

    for tid in TUMOUR_IDS:
        post = load_posterior(tid)
        log_phi = post["fission_rate"].values
        weights = post["weight"].values
        weights = weights / weights.sum()

        if tid not in sizes:
            print(f"No size for tumour {tid}, skipping")
            continue
        diameter = sizes[tid]
        N = estimate_n_glands(diameter)

        phi = 10**log_phi
        Phi = phi * K  # per-gland fission rate
        t_div = np.log(N) / Phi

        median = weighted_quantile(t_div, weights, 0.5)
        ci_lo = weighted_quantile(t_div, weights, 0.05)
        ci_hi = weighted_quantile(t_div, weights, 0.95)
        w_mean = np.average(t_div, weights=weights)

        results[tid] = {
            "samples": t_div,
            "weights": weights,
            "median": median,
            "mean": w_mean,
            "ci_90": (ci_lo, ci_hi),
            "N_glands": N,
            "diameter_cm": diameter,
            "log_phi_median": weighted_quantile(log_phi, weights, 0.5),
        }

        print(f"{tid}: diameter={diameter:.1f} cm, N={N:.1e}, "
              f"mitotic age={median:.0f} div "
              f"(90% CI: {ci_lo:.0f}-{ci_hi:.0f})")

    return results


def compute_gland_polarization(tumour_id):
    """Mean polarisation |2*beta - 1| of each gland in a tumour."""
    df = load_gland_data(tumour_id)
    polarizations = {}
    for col in df.columns:
        beta = df[col].values
        pol = np.mean(np.abs(2 * beta - 1))
        polarizations[col] = pol
    return polarizations


def compute_entropy_mitotic_ages():
    """
    Per-gland mitotic age from polarisation, using
    P(t) = P(0) * (1 - gamma_m - gamma_d)^t, with P(0) taken as the
    most polarised gland in the cohort.
    """
    all_pols = {}
    for tid in TUMOUR_IDS:
        pols = compute_gland_polarization(tid)
        all_pols[tid] = pols

    all_pol_values = [p for pols in all_pols.values() for p in pols.values()]
    P0 = max(all_pol_values)
    print(f"P(0) = {P0:.4f} (max observed polarisation)")

    results = {}
    for tid in TUMOUR_IDS:
        post = load_posterior(tid)
        weights = post["weight"].values
        weights = weights / weights.sum()

        log_gm = post["meth_rate"].values
        log_gd = post["demeth_rate"].values
        gm_median = 10**weighted_quantile(log_gm, weights, 0.5)
        gd_median = 10**weighted_quantile(log_gd, weights, 0.5)

        gamma_sum = gm_median + gd_median
        decay_rate = np.log(1 - gamma_sum)  # negative

        gland_ages = {}
        for gland_name, pol in all_pols[tid].items():
            if pol >= P0:
                t_est = 0.0
            elif pol <= 0:
                t_est = np.nan
            else:
                t_est = np.log(pol / P0) / decay_rate
            gland_ages[gland_name] = {
                "polarization": pol,
                "mitotic_age_div": t_est,
            }

        mean_pol = np.mean(list(all_pols[tid].values()))
        mean_age = np.nanmean([v["mitotic_age_div"]
                               for v in gland_ages.values()])

        results[tid] = {
            "glands": gland_ages,
            "gamma_meth": gm_median,
            "gamma_demeth": gd_median,
            "mean_polarization": mean_pol,
            "mean_mitotic_age": mean_age,
        }

        print(f"{tid}: gamma_m={gm_median:.4f}, gamma_d={gd_median:.4f}, "
              f"mean_pol={mean_pol:.3f}, "
              f"mean_age={mean_age:.0f} div")

    return results, P0


def plot_mitotic_age_posteriors(model_results):
    """Ridgeline plot of mitotic age posteriors, with implied prior in grey."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sorted_ids = sorted(model_results.keys(),
                        key=lambda t: model_results[t]["median"])

    # implied prior on mitotic age from phi ~ 10^U(-3.3, -1.0)
    rng = np.random.default_rng(0)
    n_prior = 50000

    for i, tid in enumerate(sorted_ids):
        r = model_results[tid]
        N = r["N_glands"]

        log_phi_prior = rng.uniform(-3.3, -1.0, size=n_prior)
        phi_prior = 10**log_phi_prior
        t_prior = np.log(N) / (phi_prior * K)

        x_max = max(r["samples"].max(), np.percentile(t_prior, 99), 200)
        bins = np.linspace(0, x_max, 80)
        centers = (bins[:-1] + bins[1:]) / 2

        hist_prior, _ = np.histogram(t_prior, bins=bins, density=True)
        if hist_prior.max() > 0:
            hp_scaled = hist_prior / hist_prior.max() * 0.8
            ax.fill_between(centers, i, i + hp_scaled, alpha=0.15,
                            color="grey", zorder=1)
            ax.plot(centers, i + hp_scaled, color="grey", lw=0.5,
                    ls="--", alpha=0.5, zorder=1)

        samples = r["samples"]
        weights = r["weights"]
        hist, _ = np.histogram(samples, bins=bins, weights=weights,
                               density=True)
        hist_scaled = hist / hist.max() * 0.8
        fill_color = ADENOMA_COLOR if tid in ADENOMAS else CARCINOMA_COLOR
        ax.fill_between(centers, i, i + hist_scaled, alpha=0.6,
                        color=fill_color, zorder=2)
        ax.plot(centers, i + hist_scaled, color="black", linewidth=0.5,
                zorder=2)

        median = r["median"]
        ci_lo, ci_hi = r["ci_90"]
        ax.plot([median], [i], "ko", markersize=4, zorder=3)
        ax.plot([ci_lo, ci_hi], [i, i], "k-", linewidth=1.5, zorder=3)

    ax.set_yticks([i + 0.15 for i in range(len(sorted_ids))])
    ax.set_yticklabels(sorted_ids, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("Mitotic age (cell divisions)", fontsize=12)
    ax.set_xlim(left=0)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CARCINOMA_COLOR, alpha=0.6,
                             label='Carcinoma'),
                       Patch(facecolor=ADENOMA_COLOR, alpha=0.6,
                             label='Adenoma')]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right',
              frameon=True, framealpha=0.9)

    ax2 = ax.twiny()
    ax2.set_xlim([x / 52 for x in ax.get_xlim()])  # 52 divisions per year
    ax2.set_xlabel("Approximate time (years, assuming 1 CSC div/week)",
                   fontsize=9, color="gray")
    ax2.tick_params(colors="gray", labelsize=8)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_mitotic_age_posteriors.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_mitotic_age_posteriors.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'fig_mitotic_age_posteriors.pdf'}")


def plot_entropy_vs_model_age(model_results, entropy_results):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5),
                             constrained_layout=True)

    ax = axes[0]
    tids = [t for t in TUMOUR_IDS if t in model_results and t in entropy_results]
    model_ages = [model_results[t]["median"] for t in tids]
    mean_pols = [entropy_results[t]["mean_polarization"] for t in tids]

    scatter_c = [ADENOMA_COLOR if t in ADENOMAS else CARCINOMA_COLOR for t in tids]
    ax.scatter(model_ages, mean_pols, s=60, c=scatter_c, edgecolors='black',
               lw=0.5, zorder=5)
    for i, tid in enumerate(tids):
        ax.annotate(tid, (model_ages[i], mean_pols[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9,
                    fontweight='bold')

    rho, p = stats.spearmanr(model_ages, mean_pols)
    ax.set_xlabel("Model-based mitotic age (cell divisions)", fontsize=10)
    ax.set_ylabel("Mean gland polarization |2β−1|", fontsize=10)
    ax.set_title(f"Spearman ρ = {rho:.2f}, p = {p:.3f}", fontsize=9)
    ax.set_title('A', loc='left', fontweight='bold', fontsize=12)
    ax.margins(0.1)
    ax.tick_params(labelsize=9)

    ax = axes[1]
    entropy_ages = [entropy_results[t]["mean_mitotic_age"] for t in tids]

    ax.scatter(model_ages, entropy_ages, s=60, c=scatter_c, edgecolors='black',
               lw=0.5, zorder=5)
    for i, tid in enumerate(tids):
        ax.annotate(tid, (model_ages[i], entropy_ages[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9,
                    fontweight='bold')

    finite = [(m, e) for m, e in zip(model_ages, entropy_ages)
              if np.isfinite(e)]
    if len(finite) >= 3:
        rho2, p2 = stats.spearmanr(*zip(*finite))
        ax.set_title(f"Spearman ρ = {rho2:.2f}, p = {p2:.3f}", fontsize=9)
    ax.set_xlabel("Model-based mitotic age (cell divisions)", fontsize=10)
    ax.set_ylabel("Polarisation-based mitotic age (cell divisions)", fontsize=10)
    ax.set_title('B', loc='left', fontweight='bold', fontsize=12)
    ax.margins(0.1)
    ax.tick_params(labelsize=9)

    lims = [0, max(max(model_ages), max(e for e in entropy_ages
                                         if np.isfinite(e)))]
    ax.plot(lims, lims, "k--", alpha=0.3, label="x = y")
    from matplotlib.lines import Line2D
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=CARCINOMA_COLOR, markeredgecolor='black',
                       markersize=8, label='Carcinoma'),
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ADENOMA_COLOR, markeredgecolor='black',
                       markersize=8, label='Adenoma')]
    ax.legend(handles=handles, fontsize=8)

    fig.savefig(FIGURE_DIR / "fig_mitotic_age_comparison.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_mitotic_age_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'fig_mitotic_age_comparison.pdf'}")


def plot_calendar_time_ranges(model_results):
    """Mitotic age converted to years under different CSC division rates."""
    fig, ax = plt.subplots(figsize=(8, 6))

    csc_rates = {
        "1/week": 52,
        "1/2 weeks": 26,
        "1/month": 12,
    }
    colors = {"1/week": "#2196F3", "1/2 weeks": "#FF9800", "1/month": "#4CAF50"}

    sorted_ids = sorted(model_results.keys(),
                        key=lambda t: model_results[t]["median"])

    for j, (rate_label, divs_per_year) in enumerate(csc_rates.items()):
        for i, tid in enumerate(sorted_ids):
            r = model_results[tid]
            median_yr = r["median"] / divs_per_year
            ci_lo_yr = r["ci_90"][0] / divs_per_year
            ci_hi_yr = r["ci_90"][1] / divs_per_year

            offset = (j - 1) * 0.25
            ax.plot([ci_lo_yr, ci_hi_yr], [i + offset, i + offset],
                    color=colors[rate_label], linewidth=1.5, alpha=0.7)
            ax.plot(median_yr, i + offset, "o",
                    color=colors[rate_label], markersize=5,
                    label=rate_label if i == 0 else None)

    ax.set_yticks(range(len(sorted_ids)))
    ax.set_yticklabels(sorted_ids)
    ax.set_xlabel("Estimated tumour age (years)")
    ax.set_title("Calendar time estimates under different CSC division rates")
    ax.legend(loc="lower right", fontsize=9)
    ax.axvline(x=0, color="gray", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_calendar_time_ranges.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig_calendar_time_ranges.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / 'fig_calendar_time_ranges.pdf'}")


def main():
    sizes = load_tumour_sizes()

    print("Model-based mitotic age (from fission rate)")
    model_results = compute_mitotic_age_posteriors(sizes)

    print("Polarisation-based mitotic age")
    entropy_results, P0 = compute_entropy_mitotic_ages()

    plot_mitotic_age_posteriors(model_results)
    plot_entropy_vs_model_age(model_results, entropy_results)
    plot_calendar_time_ranges(model_results)

    summary = {
        "gland_volume_mm3": GLAND_VOLUME_MM3,
        "K": K,
        "P0_estimate": P0,
        "tumours": {},
    }
    for tid in TUMOUR_IDS:
        entry = {}
        if tid in model_results:
            r = model_results[tid]
            entry["model"] = {
                "diameter_cm": r["diameter_cm"],
                "N_glands": r["N_glands"],
                "mitotic_age_median": r["median"],
                "mitotic_age_ci90": list(r["ci_90"]),
                "log_phi_median": r["log_phi_median"],
            }
        if tid in entropy_results:
            e = entropy_results[tid]
            entry["entropy"] = {
                "gamma_meth": e["gamma_meth"],
                "gamma_demeth": e["gamma_demeth"],
                "mean_polarization": e["mean_polarization"],
                "mean_mitotic_age": e["mean_mitotic_age"],
            }
        summary["tumours"][tid] = entry

    out_path = OUTPUT_DIR / "mitotic_age_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

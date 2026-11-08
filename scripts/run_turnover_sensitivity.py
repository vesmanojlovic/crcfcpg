#!/usr/bin/env python3
"""Run ABC-SMC inference for one tumour at several fixed turnover fractions in parallel."""

import os
import sys
import json
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]

import numpy as np
import pyabc
from methabc.distance import total_distance, compute_deme_matrix
from methabc.utils import import_data, write_config
from methabc.simulate import simulate


def make_log_model_with_turnover(turnover_fraction):
    """Log-scale model as in log_model.py, with turnover fixed to turnover_fraction."""
    def log_model_turnover(log_params):
        # s_driver_birth is not on log scale
        params = {k: 10**v for k, v in log_params.items()
                  if k != "s_driver_birth"}
        params["s_driver_birth"] = log_params["s_driver_birth"]
        # write_config substitutes it into the template
        params["turnover"] = turnover_fraction
        res = simulate(params)
        return {"data": res}
    return log_model_turnover


def run_single_inference(tumour_id, turnover_fraction, output_dir,
                         n_particles, max_generations, min_accept_rate):
    import pyabc

    label = f"{tumour_id}_t{turnover_fraction:.2f}"
    print(f"[{label}] Starting inference "
          f"(particles={n_particles}, max_gen={max_generations})")

    # same prior as log_model.py
    unif_params = {
        "meth_rate": (-4, -2),
        "demeth_rate": (-4, -2),
        "init_migration_rate": (-3.3, -1),
        "mu_driver_birth": (-5, -2),
        "s_driver_birth": (0, 0.2),
    }
    prior = pyabc.Distribution(
        **{key: pyabc.RV("uniform", a, b - a)
           for key, (a, b) in unif_params.items()}
    )

    data_path = REPO_DIR / "data" / "individual" / f"tumour_{tumour_id}.csv"
    observation = import_data(data_path)
    observed_matrix = compute_deme_matrix(observation)

    simulator = make_log_model_with_turnover(turnover_fraction)

    # one core per turnover fraction
    abc = pyabc.ABCSMC(
        simulator,
        prior,
        total_distance,
        population_size=n_particles,
        sampler=pyabc.sampler.SingleCoreSampler(),
    )

    db_path = Path(output_dir) / f"inference_{label}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    abc.new(
        db=f"sqlite:///{db_path}",
        observed_sum_stat={"data": observation},
        meta_info={
            "initial_dist_matrix": observed_matrix,
            "turnover_fraction": turnover_fraction,
        },
    )

    history = abc.run(
        max_nr_populations=max_generations,
        minimum_epsilon=0,
        min_acceptance_rate=min_accept_rate,
    )

    n_gen = history.max_t + 1
    print(f"[{label}] Done: {n_gen} generations, "
          f"final epsilon={history.get_all_populations()['epsilon'].iloc[-1]:.4f}")
    print(f"[{label}] Database: {db_path}")

    return str(db_path)


def _run_wrapper(args):
    return run_single_inference(*args)


def main():
    parser = ArgumentParser(
        description="Run turnover sensitivity analysis for a single tumour"
    )
    parser.add_argument(
        "--tumour", "-t", type=str, required=True,
        help="Tumour ID (e.g. E, I, J, M)"
    )
    parser.add_argument(
        "--fractions", "-f", nargs="+", type=float,
        default=[0.0, 0.15, 0.3, 0.5],
        help="Turnover fractions to test (default: 0.0 0.15 0.3 0.5)"
    )
    parser.add_argument(
        "--particles", "-p", type=int, default=1000,
        help="Number of ABC particles per generation (default: 1000)"
    )
    parser.add_argument(
        "--max-gen", "-g", type=int, default=15,
        help="Maximum number of ABC-SMC generations (default: 15)"
    )
    parser.add_argument(
        "--min-accept", type=float, default=0.03,
        help="Minimum acceptance rate to stop (default: 0.03)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str,
        default=str(REPO_DIR / "outputs" / "turnover_sensitivity"),
        help="Output directory for databases"
    )
    args = parser.parse_args()

    n_fractions = len(args.fractions)
    print(f"Tumour {args.tumour}, fractions {args.fractions}, "
          f"{args.particles} particles, max {args.max_gen} generations, "
          f"{n_fractions} processes")

    job_args = [
        (args.tumour, frac, args.output_dir,
         args.particles, args.max_gen, args.min_accept)
        for frac in args.fractions
    ]

    with mp.Pool(processes=n_fractions) as pool:
        db_paths = pool.map(_run_wrapper, job_args)

    for frac, db in zip(args.fractions, db_paths):
        print(f"turnover={frac:.2f}: {db}")

    summary = {
        "tumour": args.tumour,
        "fractions": args.fractions,
        "particles": args.particles,
        "max_generations": args.max_gen,
        "databases": dict(zip([f"{f:.2f}" for f in args.fractions], db_paths)),
    }
    summary_path = Path(args.output_dir) / f"run_summary_{args.tumour}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

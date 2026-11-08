# crcfcpg
Repo for the CRC multi-site methylation array data.

## Installation
Install [methABC](https://github.com/vesmanojlovic/methABC) first (including compiling methdemon), following its README. The scripts here use methabc's copy of methdemon and its config template. Then, in the same environment
```
git clone https://github.com/vesmanojlovic/crcfcpg
cd crcfcpg
pip install -r requirements.txt
```

## Data
- `data/individual/`: fCpG beta values for each tumour (one column per gland)
- `data/patient_info.csv`: patient metadata
- `data/all_CRC_tcga_filter.csv`, `data/TCGA-COAD_CRCspecific_fcpgs.tsv`: TCGA fCpG selection

The full arrays (`data/full_arrays/`) are not included in the repo. `scripts/filter_arrays_with_bulk.py` uses them to make `data/arrays_with_bulk/`.

## Figures and analyses
All scripts can be run from anywhere, e.g. `python scripts/data_figure.py`. Figures are written to `figures/` and other outputs to `outputs/`.

Inference results:
- `extract_posteriors.py --db_dir <dir>`: reads the inference databases in `<dir>`, writes the posterior samples and summaries to `outputs/`, and makes the posterior figures (Fig 3). Most of the scripts below need its outputs.
- `posterior_predictive_checks.py --db_dir <dir>`: posterior predictive checks
- `mitotic_age.py`: mitotic ages and calendar time ranges
- `metadata_correlation.py`: correlation of inferred parameters with patient metadata

Data and model figures:
- `data_figure.py`: data figure
- `model_data_figure.py`: simulated version of the data figure (tumour U posterior medians)
- `all_data_figures.py`, `final_tumour_figure.py`: data and simulated figures for each tumour
- `within_between_test.py`: within- vs between-side gland distances
- `generate_missing_figures.py [--ground_truth_db <db>]`: distance matrix heatmaps, and the synthetic recovery figures if the recovery database is given

Sensitivity and recovery tests:
- `parameter_sensitivity_figure.py`: illustrative simulations for each parameter
- `turnover_sensitivity.py`, `k_sweep_sensitivity.py`: simulated gland distances under different turnover and carrying capacity
- `run_turnover_sensitivity.py`: inference at different turnover fractions
- `selection_recovery_test.py`, `fixed_init_recovery_test.py`: synthetic recovery tests

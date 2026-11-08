#!/usr/bin/env python3
"""Filter full arrays to the 1164 fCpG loci, keeping glands plus bulk A/B columns."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FULL_DIR = DATA_DIR / "full_arrays"
INDIV_DIR = DATA_DIR / "individual"
OUT_DIR = DATA_DIR / "arrays_with_bulk"

TUMOUR_IDS = ["D", "E", "F", "I", "J", "M", "P", "S", "U", "X"]


def main():
    OUT_DIR.mkdir(exist_ok=True)

    for tid in TUMOUR_IDS:
        indiv = pd.read_csv(INDIV_DIR / f"tumour_{tid}.csv", usecols=["Labels"])
        keep_labels = set(indiv["Labels"].values)
        print(f"Tumour {tid}: {len(keep_labels)} target loci")

        full = pd.read_csv(FULL_DIR / f"{tid}_norm.csv", index_col=0)
        full_filtered = full[full["Labels"].isin(keep_labels)].copy()

        all_cols = full_filtered.columns.tolist()
        # bulk columns are the two not in the individual file (naming varies)
        indiv_gland_cols = set(
            pd.read_csv(INDIV_DIR / f"tumour_{tid}.csv", nrows=0).columns
        ) - {"Labels"}
        bulk_cols = [c for c in all_cols
                     if c != "Labels" and c not in indiv_gland_cols]
        assert len(bulk_cols) == 2, \
            f"Expected 2 bulk columns for {tid}, got {bulk_cols}"
        bulk_a_col, bulk_b_col = bulk_cols[0], bulk_cols[1]

        full_filtered = full_filtered.reset_index(drop=True)
        full_filtered = full_filtered.set_index("Labels")

        out_path = OUT_DIR / f"tumour_{tid}.csv"
        full_filtered.to_csv(out_path)

        n_glands = len(full_filtered.columns) - 2  # minus bulk A and B
        print(f"  -> {len(full_filtered)} loci, {n_glands} glands + "
              f"{bulk_a_col}, {bulk_b_col}")

    print(f"Saved files to {OUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare a kMoL-ready CSV from the finalized UmamiTaste training datasets.

The script auto-detects the SMILES column (preferring ml_SMILES / parent_SMILES /
standardized_SMILES) and a binary target column (preferring label), validates all
structures with RDKit, removes exact duplicate SMILES-label rows, reports label
conflicts, and writes the minimal CSV expected by kMoL:

    smiles,umami

Examples
--------
python src/06_prepare_kmol_input.py \
  --input data/model_sets/training_full_primary.csv \
  --output data/kmol/umami_full_primary.csv

python src/06_prepare_kmol_input.py \
  --input data/model_sets/training_strict_tasteless.csv \
  --output data/kmol/umami_strict.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from rdkit import Chem

SMILES_CANDIDATES = [
    "ml_SMILES",
    "parent_SMILES",
    "standardized_SMILES",
    "canonical_SMILES",
    "smiles",
    "SMILES",
]
LABEL_CANDIDATES = ["label", "umami", "target", "y"]


def choose_column(columns, candidates, explicit=None):
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Column '{explicit}' not found. Available: {list(columns)}")
        return explicit
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"Could not auto-detect column. Tried: {candidates}. Available: {list(columns)}")


def canonicalize_smiles(s: str) -> str | None:
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input training CSV")
    ap.add_argument("--output", required=True, help="Output kMoL CSV")
    ap.add_argument("--smiles-column", default=None)
    ap.add_argument("--label-column", default=None)
    ap.add_argument(
        "--drop-conflicts",
        action="store_true",
        help="Drop structures occurring with both labels instead of failing.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    smiles_col = choose_column(df.columns, SMILES_CANDIDATES, args.smiles_column)
    label_col = choose_column(df.columns, LABEL_CANDIDATES, args.label_column)

    work = df[[smiles_col, label_col]].copy()
    work.columns = ["smiles_original", "umami"]

    # Force binary integer target.
    numeric = pd.to_numeric(work["umami"], errors="coerce")
    if numeric.isna().any():
        bad = work.loc[numeric.isna(), "umami"].drop_duplicates().tolist()[:10]
        raise ValueError(f"Non-numeric labels found, examples: {bad}")
    work["umami"] = numeric.astype(float)
    invalid_labels = sorted(set(work["umami"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Labels must be 0/1. Found: {invalid_labels}")

    work["smiles"] = work["smiles_original"].map(canonicalize_smiles)
    invalid = work[work["smiles"].isna()].copy()
    valid = work[work["smiles"].notna()].copy()

    # Detect conflicting labels after canonicalization.
    nunique = valid.groupby("smiles")["umami"].nunique()
    conflict_smiles = set(nunique[nunique > 1].index)
    conflicts = valid[valid["smiles"].isin(conflict_smiles)].sort_values(["smiles", "umami"])

    audit_dir = out_path.parent / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    invalid.to_csv(audit_dir / f"{out_path.stem}_invalid_smiles.csv", index=False)
    conflicts.to_csv(audit_dir / f"{out_path.stem}_label_conflicts.csv", index=False)

    if conflict_smiles:
        if args.drop_conflicts:
            valid = valid[~valid["smiles"].isin(conflict_smiles)].copy()
        else:
            print(
                f"ERROR: {len(conflict_smiles)} canonical structure(s) have conflicting labels. "
                f"See {audit_dir / (out_path.stem + '_label_conflicts.csv')}",
                file=sys.stderr,
            )
            sys.exit(2)

    before = len(valid)
    valid = valid.drop_duplicates(subset=["smiles", "umami"], keep="first")
    duplicate_rows_removed = before - len(valid)

    out = valid[["smiles", "umami"]].reset_index(drop=True)
    out.to_csv(out_path, index=False)

    counts = out["umami"].value_counts().sort_index().to_dict()
    summary = pd.DataFrame(
        [
            {"metric": "input_rows", "value": len(df)},
            {"metric": "invalid_smiles", "value": len(invalid)},
            {"metric": "conflicting_canonical_structures", "value": len(conflict_smiles)},
            {"metric": "duplicate_rows_removed", "value": duplicate_rows_removed},
            {"metric": "final_rows", "value": len(out)},
            {"metric": "negative_n", "value": counts.get(0, 0)},
            {"metric": "umami_n", "value": counts.get(1, 0)},
        ]
    )
    summary.to_csv(audit_dir / f"{out_path.stem}_summary.csv", index=False)

    print(f"Input:       {in_path}")
    print(f"SMILES col:  {smiles_col}")
    print(f"Label col:   {label_col}")
    print(f"Output:      {out_path}")
    print(f"Rows:        {len(out)} (negative={counts.get(0, 0)}, umami={counts.get(1, 0)})")
    print(f"Invalid:     {len(invalid)}")
    print(f"Conflicts:   {len(conflict_smiles)}")
    print(f"Duplicates:  {duplicate_rows_removed} removed")


if __name__ == "__main__":
    main()

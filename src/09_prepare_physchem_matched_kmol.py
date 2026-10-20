#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem


def pick_smiles_col(df):
    for c in ["ml_SMILES","smiles","SMILES","canonical_SMILES","parent_SMILES","standardized_SMILES"]:
        if c in df.columns:
            return c
    raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")


def pick_label_col(df):
    for c in ["label","umami","target"]:
        if c in df.columns:
            return c
    raise ValueError(f"No label column found. Columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    smi_col = pick_smiles_col(df)
    lab_col = pick_label_col(df)

    rows = []
    invalid = []
    for i, r in df.iterrows():
        smi = str(r[smi_col]).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append((i, smi))
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        lab = float(r[lab_col])
        if lab not in (0.0, 1.0):
            raise ValueError(f"Non-binary label at row {i}: {lab}")
        rows.append({"smiles": can, "umami": lab})

    out_df = pd.DataFrame(rows)
    dup = out_df.duplicated(subset=["smiles"], keep=False)
    if dup.any():
        conflict = out_df[dup].groupby("smiles")["umami"].nunique()
        bad = conflict[conflict > 1]
        if len(bad):
            raise ValueError(f"Label conflicts after canonicalization: {len(bad)}")
        out_df = out_df.drop_duplicates(subset=["smiles"], keep="first").reset_index(drop=True)

    out_df.to_csv(out, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Valid unique molecules: {len(out_df)}")
    print(f"Positive: {(out_df.umami==1).sum()}")
    print(f"Negative: {(out_df.umami==0).sum()}")
    print(f"Invalid SMILES: {len(invalid)}")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem


def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of these columns found: {names}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--weight-output', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    smi_col = pick_col(df, ['ml_SMILES','smiles','canonical_SMILES','parent_SMILES'])
    lab_col = pick_col(df, ['label','umami'])

    out = pd.DataFrame()
    out['smiles'] = df[smi_col].astype(str)
    out['umami'] = pd.to_numeric(df[lab_col], errors='raise').astype(float)

    invalid = []
    canon = []
    for i, s in enumerate(out['smiles']):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            invalid.append((i,s))
            canon.append(None)
        else:
            canon.append(Chem.MolToSmiles(mol, canonical=True))
    if invalid:
        raise ValueError(f"Invalid SMILES found: {invalid[:10]}")
    out['smiles'] = canon

    if not set(out['umami'].unique()).issubset({0.0,1.0}):
        raise ValueError(f"Labels must be 0/1: {sorted(out['umami'].unique())}")

    conflicts = out.groupby('smiles')['umami'].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(f"Label conflicts after canonicalization: {len(conflicts)}")

    if out['smiles'].duplicated().any():
        d = int(out['smiles'].duplicated(keep=False).sum())
        raise ValueError(f"Duplicate canonical structures found: {d} rows")

    n_pos = int((out['umami'] == 1.0).sum())
    n_neg = int((out['umami'] == 0.0).sum())
    pos_weight = n_neg / n_pos

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    import json
    Path(args.weight_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.weight_output, 'w') as f:
        json.dump({'positive':n_pos,'negative':n_neg,'pos_weight':pos_weight}, f, indent=2)

    print(f"Rows: {len(out)}")
    print(f"Positive: {n_pos}")
    print(f"Negative: {n_neg}")
    print(f"pos_weight: {pos_weight:.6f}")
    print(f"Saved: {args.output}")
    print(f"Saved: {args.weight_output}")


if __name__ == '__main__':
    main()

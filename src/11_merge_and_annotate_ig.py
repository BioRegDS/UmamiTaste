#!/usr/bin/env python3
"""
11_merge_and_annotate_ig.py

Merge out-of-fold atom-level Integrated Gradients (IG) outputs from all folds,
identify umami-positive molecules, and annotate atoms with chemically meaningful
functional groups using RDKit SMARTS.

Designed for files produced by:
  10_export_signed_ig_v2.py

Typical usage:
  python src/11_merge_and_annotate_ig.py \
    --ig-dir data/ig/full_primary_oof_v2 \
    --positive data/processed/umami_positive_final.csv \
    --output-dir data/ig/analysis
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

FUNCTIONAL_GROUP_SMARTS = {
    "carboxylic_acid": "C(=O)[OX2H1]",
    "carboxylate": "C(=O)[O-]",
    "amide": "C(=O)N",
    "primary_amine": "[NX3;H2;!$(NC=O)]",
    "secondary_amine": "[NX3;H1;!$(NC=O)]",
    "tertiary_amine": "[NX3;H0;!$(NC=O);!$([N+])]",
    "quaternary_ammonium": "[N+;X4]",
    "hydroxyl": "[OX2H;!$([OX2H]C=O)]",
    "phenol": "[OX2H]-c",
    "ether": "[OD2]([#6])[#6]",
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "ester": "C(=O)O[#6]",
    "phosphate": "P(=O)(O)(O)",
    "phosphonate": "[#6]P(=O)(O)O",
    "sulfate": "S(=O)(=O)(O)O",
    "sulfonamide": "S(=O)(=O)N",
    "thiol": "[SX2H]",
    "thioether": "[#16X2]([#6])[#6]",
    "guanidino": "NC(=N)N",
    "urea": "NC(=O)N",
    "imidazole": "c1ncc[nH]1",
    "purine_like": "n1cnc2ncnc12",
    "pyrimidine_like": "n1ccnc(=O)n1",
    "aromatic_ring_atom": "[a]",
    "heteroaromatic_atom": "[a;!#6]",
    "aliphatic_nitrogen": "[N;!a]",
    "aliphatic_oxygen": "[O;!a]",
}


def compile_patterns():
    patterns = {}
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Invalid SMARTS: {name} -> {smarts}")
        patterns[name] = patt
    return patterns


def canonical_smiles(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_positive_smiles(path):
    df = pd.read_csv(path)
    candidate_cols = [
        "ml_SMILES", "parent_SMILES", "standardized_SMILES",
        "canonical_SMILES", "smiles", "SMILES"
    ]
    smi_col = next((c for c in candidate_cols if c in df.columns), None)
    if smi_col is None:
        raise ValueError(f"No SMILES column found in positive file: {list(df.columns)}")
    df["_canonical"] = df[smi_col].map(canonical_smiles)
    if df["_canonical"].isna().any():
        raise ValueError("Invalid SMILES found in positive dataset")
    return set(df["_canonical"]), smi_col


def merge_files(ig_dir, pattern):
    paths = sorted(glob.glob(str(Path(ig_dir) / pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched {Path(ig_dir) / pattern}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["_source_file"] = str(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), paths


def annotate_molecule(smiles, patterns):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES during annotation: {smiles}")
    atom_groups = defaultdict(set)
    for group_name, patt in patterns.items():
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            for atom_idx in match:
                atom_groups[int(atom_idx)].add(group_name)
    return mol, atom_groups


def summarize_functional_groups(long_df):
    rows = []
    for group_name, g in long_df.groupby("functional_group", sort=True):
        mol_signed = g.groupby(["fold", "dataset_id", "canonical_smiles"])["signed_ig"].sum().values
        mol_abs = g.groupby(["fold", "dataset_id", "canonical_smiles"])["abs_ig"].sum().values
        atom_signed = g["signed_ig"].values
        atom_abs = g["abs_ig"].values
        rows.append({
            "functional_group": group_name,
            "n_atom_rows": len(g),
            "n_unique_molecules": g["canonical_smiles"].nunique(),
            "n_unique_atoms": g[["canonical_smiles", "atom_index"]].drop_duplicates().shape[0],
            "mean_atom_signed_ig": float(np.mean(atom_signed)),
            "median_atom_signed_ig": float(np.median(atom_signed)),
            "mean_atom_abs_ig": float(np.mean(atom_abs)),
            "median_atom_abs_ig": float(np.median(atom_abs)),
            "mean_molecule_group_signed_ig": float(np.mean(mol_signed)),
            "median_molecule_group_signed_ig": float(np.median(mol_signed)),
            "mean_molecule_group_abs_ig": float(np.mean(mol_abs)),
            "median_molecule_group_abs_ig": float(np.median(mol_abs)),
            "fraction_positive_atom_signed_ig": float(np.mean(atom_signed > 0)),
            "fraction_negative_atom_signed_ig": float(np.mean(atom_signed < 0)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["median_molecule_group_abs_ig", "n_unique_molecules"], ascending=[False, False])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ig-dir", required=True)
    p.add_argument("--positive", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    atom_df, atom_paths = merge_files(args.ig_dir, "fold_*_atom_ig.csv")
    mol_df, mol_paths = merge_files(args.ig_dir, "fold_*_molecule_ig_summary.csv")

    atom_df["canonical_smiles"] = atom_df["smiles"].map(canonical_smiles)
    mol_df["canonical_smiles"] = mol_df["smiles"].map(canonical_smiles)
    if atom_df["canonical_smiles"].isna().any() or mol_df["canonical_smiles"].isna().any():
        raise ValueError("Invalid SMILES found in IG outputs")

    positive_set, positive_smiles_col = load_positive_smiles(args.positive)
    atom_df["is_umami_positive"] = atom_df["canonical_smiles"].isin(positive_set).astype(int)
    mol_df["is_umami_positive"] = mol_df["canonical_smiles"].isin(positive_set).astype(int)

    atom_df.to_csv(outdir / "oof_atom_ig_master.csv", index=False)
    mol_df.to_csv(outdir / "oof_molecule_ig_master.csv", index=False)

    umami_atom = atom_df[atom_df["is_umami_positive"] == 1].copy()
    patterns = compile_patterns()
    cache = {}
    fg_labels = []
    fg_counts = []
    long_rows = []

    for _, row in umami_atom.iterrows():
        smi = row["canonical_smiles"]
        if smi not in cache:
            cache[smi] = annotate_molecule(smi, patterns)
        mol, atom_groups = cache[smi]
        atom_idx = int(row["atom_index"])
        if atom_idx >= mol.GetNumAtoms():
            raise ValueError(f"Atom index mismatch for {smi}: {atom_idx} >= {mol.GetNumAtoms()}")
        groups = sorted(atom_groups.get(atom_idx, set()))
        fg_labels.append(";".join(groups) if groups else "unassigned")
        fg_counts.append(len(groups))
        base = row.to_dict()
        if groups:
            for gname in groups:
                rec = dict(base)
                rec["functional_group"] = gname
                rec["n_groups_for_atom"] = len(groups)
                long_rows.append(rec)
        else:
            rec = dict(base)
            rec["functional_group"] = "unassigned"
            rec["n_groups_for_atom"] = 0
            long_rows.append(rec)

    umami_atom["functional_groups"] = fg_labels
    umami_atom["n_functional_groups"] = fg_counts
    umami_atom.to_csv(outdir / "umami_atom_ig_annotated.csv", index=False)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(outdir / "umami_functional_group_long.csv", index=False)

    fg_summary = summarize_functional_groups(long_df)
    fg_summary.to_csv(outdir / "umami_functional_group_summary.csv", index=False)

    if not long_df.empty:
        per_mol = (
            long_df.groupby(["fold", "dataset_id", "canonical_smiles", "functional_group"], dropna=False)
            .agg(
                n_atoms_in_group=("atom_index", "nunique"),
                signed_ig_sum=("signed_ig", "sum"),
                signed_ig_mean=("signed_ig", "mean"),
                abs_ig_sum=("abs_ig", "sum"),
                abs_ig_mean=("abs_ig", "mean"),
            )
            .reset_index()
        )
        per_mol.to_csv(outdir / "umami_functional_group_per_molecule.csv", index=False)

    summary = pd.DataFrame([
        {"item": "atom_ig_files", "value": len(atom_paths)},
        {"item": "molecule_ig_files", "value": len(mol_paths)},
        {"item": "oof_atom_rows", "value": len(atom_df)},
        {"item": "oof_unique_molecules", "value": mol_df["canonical_smiles"].nunique()},
        {"item": "positive_master_molecules", "value": len(positive_set)},
        {"item": "positive_molecules_found_in_oof", "value": mol_df.loc[mol_df["is_umami_positive"] == 1, "canonical_smiles"].nunique()},
        {"item": "positive_atom_rows", "value": len(umami_atom)},
        {"item": "positive_atoms_annotated", "value": int((umami_atom["n_functional_groups"] > 0).sum())},
        {"item": "positive_atoms_unassigned", "value": int((umami_atom["n_functional_groups"] == 0).sum())},
        {"item": "functional_group_definitions", "value": len(patterns)},
        {"item": "positive_smiles_column", "value": positive_smiles_col},
    ])
    summary.to_csv(outdir / "ig_merge_summary.csv", index=False)

    print("Done.")
    print(f"OOF molecules: {mol_df['canonical_smiles'].nunique()}")
    print(f"Umami positives found: {mol_df.loc[mol_df['is_umami_positive'] == 1, 'canonical_smiles'].nunique()} / {len(positive_set)}")
    print(f"Umami atom rows: {len(umami_atom)}")
    print(f"Annotated umami atoms: {int((umami_atom['n_functional_groups'] > 0).sum())} / {len(umami_atom)}")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()

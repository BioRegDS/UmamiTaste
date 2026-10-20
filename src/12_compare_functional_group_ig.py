#!/usr/bin/env python3
"""
12_compare_functional_group_ig.py

Annotate all OOF molecules/atoms with functional groups and compare:
  1) functional-group prevalence in umami vs non-umami molecules
  2) molecule-level signed IG among molecules containing each group
  3) molecule-level absolute IG among molecules containing each group
  4) fraction of molecules with positive group-level signed IG

Statistical tests:
  - Fisher's exact test for prevalence
  - Mann-Whitney U for group-level IG distributions
  - Benjamini-Hochberg FDR correction

Inputs:
  --atom-master data/ig/analysis/oof_atom_ig_master.csv
  --positive data/processed/umami_positive_final.csv

Outputs:
  all_atom_ig_annotated.csv
  all_functional_group_long.csv
  functional_group_per_molecule_all.csv
  functional_group_comparison.csv
  functional_group_prevalence.csv
  functional_group_ig_stats.csv
  functional_group_analysis_summary.csv
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import fisher_exact, mannwhitneyu


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
    out = {}
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            raise ValueError(f"Invalid SMARTS for {name}: {smarts}")
        out[name] = mol
    return out


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def bh_fdr(pvals):
    """
    Benjamini-Hochberg adjusted p-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)

    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def cliffs_delta(x, y):
    """
    Cliff's delta effect size:
      +1 => all x > y
      -1 => all x < y
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan

    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))


def annotate_atoms(atom_df, patterns):
    cache = {}
    labels = []
    counts = []
    long_rows = []

    for _, row in atom_df.iterrows():
        smi = row["canonical_smiles"]
        atom_idx = int(row["atom_index"])

        if smi not in cache:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")

            atom_groups = defaultdict(set)
            for group, patt in patterns.items():
                for match in mol.GetSubstructMatches(patt, uniquify=True):
                    for idx in match:
                        atom_groups[int(idx)].add(group)

            cache[smi] = (mol, atom_groups)

        mol, atom_groups = cache[smi]
        if atom_idx >= mol.GetNumAtoms():
            raise ValueError(
                f"Atom index {atom_idx} out of range for {smi}"
            )

        groups = sorted(atom_groups.get(atom_idx, set()))
        labels.append(";".join(groups) if groups else "unassigned")
        counts.append(len(groups))

        base = row.to_dict()
        if groups:
            for g in groups:
                rec = dict(base)
                rec["functional_group"] = g
                rec["n_groups_for_atom"] = len(groups)
                long_rows.append(rec)
        else:
            rec = dict(base)
            rec["functional_group"] = "unassigned"
            rec["n_groups_for_atom"] = 0
            long_rows.append(rec)

    out = atom_df.copy()
    out["functional_groups"] = labels
    out["n_functional_groups"] = counts
    long_df = pd.DataFrame(long_rows)
    return out, long_df


def prevalence_stats(per_mol_df, molecule_df):
    rows = []

    n_pos_total = int((molecule_df["is_umami_positive"] == 1).sum())
    n_neg_total = int((molecule_df["is_umami_positive"] == 0).sum())

    for group, g in per_mol_df.groupby("functional_group"):
        pos_present = int(
            g.loc[g["is_umami_positive"] == 1, "canonical_smiles"].nunique()
        )
        neg_present = int(
            g.loc[g["is_umami_positive"] == 0, "canonical_smiles"].nunique()
        )
        pos_absent = n_pos_total - pos_present
        neg_absent = n_neg_total - neg_present

        table = [[pos_present, pos_absent], [neg_present, neg_absent]]
        odds_ratio, p = fisher_exact(table, alternative="two-sided")

        rows.append({
            "functional_group": group,
            "umami_present": pos_present,
            "umami_total": n_pos_total,
            "umami_prevalence": pos_present / n_pos_total if n_pos_total else np.nan,
            "nonumami_present": neg_present,
            "nonumami_total": n_neg_total,
            "nonumami_prevalence": neg_present / n_neg_total if n_neg_total else np.nan,
            "prevalence_difference": (
                pos_present / n_pos_total - neg_present / n_neg_total
                if n_pos_total and n_neg_total else np.nan
            ),
            "fisher_odds_ratio": odds_ratio,
            "fisher_p": p,
        })

    df = pd.DataFrame(rows)
    if len(df):
        df["fisher_q"] = bh_fdr(df["fisher_p"].values)
    return df


def ig_stats(per_mol_df):
    rows = []

    for group, g in per_mol_df.groupby("functional_group"):
        pos = g[g["is_umami_positive"] == 1]
        neg = g[g["is_umami_positive"] == 0]

        rec = {
            "functional_group": group,
            "n_umami_molecules": pos["canonical_smiles"].nunique(),
            "n_nonumami_molecules": neg["canonical_smiles"].nunique(),
        }

        for col, prefix in [
            ("signed_ig_sum", "signed"),
            ("abs_ig_sum", "abs"),
        ]:
            x = pos[col].values.astype(float)
            y = neg[col].values.astype(float)

            rec[f"umami_{prefix}_median"] = (
                float(np.median(x)) if len(x) else np.nan
            )
            rec[f"nonumami_{prefix}_median"] = (
                float(np.median(y)) if len(y) else np.nan
            )
            rec[f"{prefix}_median_difference"] = (
                rec[f"umami_{prefix}_median"]
                - rec[f"nonumami_{prefix}_median"]
                if len(x) and len(y)
                else np.nan
            )

            if len(x) and len(y):
                stat, p = mannwhitneyu(x, y, alternative="two-sided")
                rec[f"{prefix}_mw_u"] = float(stat)
                rec[f"{prefix}_mw_p"] = float(p)
                rec[f"{prefix}_cliffs_delta"] = float(cliffs_delta(x, y))
            else:
                rec[f"{prefix}_mw_u"] = np.nan
                rec[f"{prefix}_mw_p"] = np.nan
                rec[f"{prefix}_cliffs_delta"] = np.nan

        if len(pos):
            rec["umami_fraction_positive_signed"] = float(
                np.mean(pos["signed_ig_sum"].values > 0)
            )
        else:
            rec["umami_fraction_positive_signed"] = np.nan

        if len(neg):
            rec["nonumami_fraction_positive_signed"] = float(
                np.mean(neg["signed_ig_sum"].values > 0)
            )
        else:
            rec["nonumami_fraction_positive_signed"] = np.nan

        rec["fraction_positive_difference"] = (
            rec["umami_fraction_positive_signed"]
            - rec["nonumami_fraction_positive_signed"]
            if not pd.isna(rec["umami_fraction_positive_signed"])
            and not pd.isna(rec["nonumami_fraction_positive_signed"])
            else np.nan
        )

        rows.append(rec)

    df = pd.DataFrame(rows)
    if len(df):
        signed_mask = df["signed_mw_p"].notna()
        abs_mask = df["abs_mw_p"].notna()

        df["signed_mw_q"] = np.nan
        df["abs_mw_q"] = np.nan

        if signed_mask.any():
            df.loc[signed_mask, "signed_mw_q"] = bh_fdr(
                df.loc[signed_mask, "signed_mw_p"].values
            )
        if abs_mask.any():
            df.loc[abs_mask, "abs_mw_q"] = bh_fdr(
                df.loc[abs_mask, "abs_mw_p"].values
            )

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atom-master", required=True)
    p.add_argument("--positive", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    atom_df = pd.read_csv(args.atom_master)
    if "canonical_smiles" not in atom_df.columns:
        atom_df["canonical_smiles"] = atom_df["smiles"].map(canonical_smiles)

    if atom_df["canonical_smiles"].isna().any():
        raise ValueError("Invalid SMILES found in atom master")

    pos_df = pd.read_csv(args.positive)
    candidate_cols = [
        "ml_SMILES",
        "parent_SMILES",
        "standardized_SMILES",
        "canonical_SMILES",
        "smiles",
        "SMILES",
    ]
    pos_col = next((c for c in candidate_cols if c in pos_df.columns), None)
    if pos_col is None:
        raise ValueError("Could not find SMILES column in positive file")

    positive_set = set(
        s for s in pos_df[pos_col].map(canonical_smiles).tolist()
        if s is not None
    )

    atom_df["is_umami_positive"] = atom_df["canonical_smiles"].isin(
        positive_set
    ).astype(int)

    # molecule-level label table
    molecule_df = (
        atom_df[
            ["fold", "dataset_id", "canonical_smiles", "is_umami_positive"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    patterns = compile_patterns()
    annotated_df, long_df = annotate_atoms(atom_df, patterns)

    annotated_df.to_csv(
        outdir / "all_atom_ig_annotated.csv",
        index=False,
    )
    long_df.to_csv(
        outdir / "all_functional_group_long.csv",
        index=False,
    )

    per_mol = (
        long_df.groupby(
            [
                "fold",
                "dataset_id",
                "canonical_smiles",
                "is_umami_positive",
                "functional_group",
            ],
            dropna=False,
        )
        .agg(
            n_atoms_in_group=("atom_index", "nunique"),
            signed_ig_sum=("signed_ig", "sum"),
            signed_ig_mean=("signed_ig", "mean"),
            abs_ig_sum=("abs_ig", "sum"),
            abs_ig_mean=("abs_ig", "mean"),
        )
        .reset_index()
    )
    per_mol.to_csv(
        outdir / "functional_group_per_molecule_all.csv",
        index=False,
    )

    prev = prevalence_stats(per_mol, molecule_df)
    prev.to_csv(
        outdir / "functional_group_prevalence.csv",
        index=False,
    )

    ig = ig_stats(per_mol)
    ig.to_csv(
        outdir / "functional_group_ig_stats.csv",
        index=False,
    )

    comparison = prev.merge(
        ig,
        on="functional_group",
        how="outer",
    )

    # Helpful ranking columns
    comparison["signed_significant_fdr05"] = (
        comparison["signed_mw_q"] < 0.05
    ).astype(int)
    comparison["prevalence_significant_fdr05"] = (
        comparison["fisher_q"] < 0.05
    ).astype(int)

    comparison = comparison.sort_values(
        [
            "signed_significant_fdr05",
            "prevalence_significant_fdr05",
            "signed_cliffs_delta",
        ],
        ascending=[False, False, False],
    )

    comparison.to_csv(
        outdir / "functional_group_comparison.csv",
        index=False,
    )

    summary = pd.DataFrame([
        {
            "item": "total_molecules",
            "value": molecule_df["canonical_smiles"].nunique(),
        },
        {
            "item": "umami_molecules",
            "value": int(
                molecule_df.loc[
                    molecule_df["is_umami_positive"] == 1,
                    "canonical_smiles",
                ].nunique()
            ),
        },
        {
            "item": "nonumami_molecules",
            "value": int(
                molecule_df.loc[
                    molecule_df["is_umami_positive"] == 0,
                    "canonical_smiles",
                ].nunique()
            ),
        },
        {
            "item": "atom_rows",
            "value": len(atom_df),
        },
        {
            "item": "functional_group_definitions",
            "value": len(patterns),
        },
        {
            "item": "functional_groups_tested",
            "value": len(comparison),
        },
    ])
    summary.to_csv(
        outdir / "functional_group_analysis_summary.csv",
        index=False,
    )

    print("Done.")
    print(f"Total molecules: {molecule_df['canonical_smiles'].nunique()}")
    print(
        "Umami molecules: "
        f"{molecule_df.loc[molecule_df['is_umami_positive'] == 1, 'canonical_smiles'].nunique()}"
    )
    print(
        "Non-umami molecules: "
        f"{molecule_df.loc[molecule_df['is_umami_positive'] == 0, 'canonical_smiles'].nunique()}"
    )
    print(f"Functional groups tested: {len(comparison)}")
    print(
        "Signed-IG significant groups (FDR<0.05): "
        f"{int((comparison['signed_mw_q'] < 0.05).sum())}"
    )
    print(
        "Prevalence significant groups (FDR<0.05): "
        f"{int((comparison['fisher_q'] < 0.05).sum())}"
    )
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()

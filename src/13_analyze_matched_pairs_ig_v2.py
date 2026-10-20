#!/usr/bin/env python3
"""
13_analyze_matched_pairs_ig_v2.py

Use PRECOMPUTED MCS / atom-index annotations from:
  data/analysis/high_similarity_pairs_with_mcs.csv

This preserves the exact pair definitions and MCS decomposition established
earlier, instead of recomputing MCS during IG analysis.

Expected pair-file columns:
  positive_smiles
  negative_smiles
  tanimoto_ecfp4
  mcs_smarts
  pos_mcs_atom_indices
  neg_mcs_atom_indices
  pos_specific_atom_indices
  neg_specific_atom_indices
  (optional) positive_id, negative_id, positive_name, negative_name

Input atom IG:
  data/ig/analysis/oof_atom_ig_master.csv

Outputs:
  matched_pair_ig_summary.csv
  matched_pair_atom_annotations.csv
  matched_pair_group_stats.csv
  matched_pair_analysis_summary.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import wilcoxon


def canonical_smiles(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def parse_index_list(value):
    """
    Parse semicolon-separated atom indices such as:
      '0;1;2;3'
    Empty / NaN -> []
    """
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s == "":
        return []
    return [int(x) for x in s.split(";") if str(x).strip() != ""]


def build_atom_lookup(atom_df):
    lookup = {}
    for smi, g in atom_df.groupby("canonical_smiles", sort=False):
        lookup[smi] = g.copy().sort_values("atom_index")
    return lookup


def region_stats(df, indices):
    indices = list(indices)

    if len(indices) == 0:
        return {
            "n_atoms": 0,
            "signed_sum": 0.0,
            "signed_mean": np.nan,
            "signed_median": np.nan,
            "abs_sum": 0.0,
            "abs_mean": np.nan,
            "abs_median": np.nan,
            "positive_fraction": np.nan,
            "negative_fraction": np.nan,
        }

    sub = df[df["atom_index"].isin(indices)].copy()

    if len(sub) != len(set(indices)):
        found = sorted(sub["atom_index"].astype(int).unique().tolist())
        expected = sorted(set(indices))
        raise ValueError(
            f"Atom-index mismatch. Expected {expected}, found {found}"
        )

    signed = sub["signed_ig"].astype(float).values
    absig = sub["abs_ig"].astype(float).values

    return {
        "n_atoms": len(sub),
        "signed_sum": float(np.sum(signed)),
        "signed_mean": float(np.mean(signed)),
        "signed_median": float(np.median(signed)),
        "abs_sum": float(np.sum(absig)),
        "abs_mean": float(np.mean(absig)),
        "abs_median": float(np.median(absig)),
        "positive_fraction": float(np.mean(signed > 0)),
        "negative_fraction": float(np.mean(signed < 0)),
    }


def prefix_dict(d, prefix):
    return {f"{prefix}_{k}": v for k, v in d.items()}


def safe_ratio(num, den):
    if den is None or np.isnan(den) or den == 0:
        return np.nan
    return num / den


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    if len(pvals) == 0:
        return np.array([])

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


def validate_indices(mol, common_indices, specific_indices, pair_id, role):
    n_atoms = mol.GetNumAtoms()
    all_idx = set(range(n_atoms))
    common = set(common_indices)
    specific = set(specific_indices)

    bad_common = sorted(i for i in common if i not in all_idx)
    bad_specific = sorted(i for i in specific if i not in all_idx)

    if bad_common or bad_specific:
        raise ValueError(
            f"Pair {pair_id} {role}: invalid atom indices. "
            f"common={bad_common}, specific={bad_specific}, n_atoms={n_atoms}"
        )

    overlap = common & specific
    if overlap:
        raise ValueError(
            f"Pair {pair_id} {role}: common/specific overlap: {sorted(overlap)}"
        )

    union = common | specific
    if union != all_idx:
        missing = sorted(all_idx - union)
        extra = sorted(union - all_idx)
        raise ValueError(
            f"Pair {pair_id} {role}: common + specific do not cover molecule. "
            f"missing={missing}, extra={extra}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--atom-master", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--positive-smiles-col", default="positive_smiles")
    p.add_argument("--negative-smiles-col", default="negative_smiles")
    p.add_argument("--similarity-col", default="tanimoto_ecfp4")
    p.add_argument("--pair-id-col", default=None)

    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs)
    atom_df = pd.read_csv(args.atom_master)

    required_pair_cols = [
        args.positive_smiles_col,
        args.negative_smiles_col,
        "mcs_smarts",
        "pos_mcs_atom_indices",
        "neg_mcs_atom_indices",
        "pos_specific_atom_indices",
        "neg_specific_atom_indices",
    ]

    missing_cols = [c for c in required_pair_cols if c not in pairs.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in pair file: {missing_cols}"
        )

    if "canonical_smiles" not in atom_df.columns:
        if "smiles" not in atom_df.columns:
            raise ValueError(
                "atom-master must contain canonical_smiles or smiles"
            )
        atom_df["canonical_smiles"] = atom_df["smiles"].map(canonical_smiles)

    if atom_df["canonical_smiles"].isna().any():
        raise ValueError("Invalid SMILES in atom master")

    lookup = build_atom_lookup(atom_df)

    summary_rows = []
    atom_rows = []
    missing_rows = []
    validation_rows = []

    for idx, row in pairs.iterrows():
        pair_id = (
            row[args.pair_id_col]
            if args.pair_id_col is not None
            else idx + 1
        )

        similarity = (
            row[args.similarity_col]
            if args.similarity_col in pairs.columns
            else np.nan
        )

        pos_id = row["positive_id"] if "positive_id" in pairs.columns else ""
        neg_id = row["negative_id"] if "negative_id" in pairs.columns else ""
        pos_name = row["positive_name"] if "positive_name" in pairs.columns else ""
        neg_name = row["negative_name"] if "negative_name" in pairs.columns else ""

        usmi_raw = row[args.positive_smiles_col]
        nsmi_raw = row[args.negative_smiles_col]

        usmi = canonical_smiles(usmi_raw)
        nsmi = canonical_smiles(nsmi_raw)

        if usmi is None or nsmi is None:
            raise ValueError(f"Pair {pair_id}: invalid SMILES")

        if usmi not in lookup or nsmi not in lookup:
            missing_rows.append({
                "pair_id": pair_id,
                "positive_id": pos_id,
                "negative_id": neg_id,
                "positive_name": pos_name,
                "negative_name": neg_name,
                "positive_smiles": usmi,
                "negative_smiles": nsmi,
                "positive_found_in_ig": int(usmi in lookup),
                "negative_found_in_ig": int(nsmi in lookup),
            })
            continue

        udf = lookup[usmi]
        ndf = lookup[nsmi]

        umol = Chem.MolFromSmiles(usmi)
        nmol = Chem.MolFromSmiles(nsmi)

        u_common = parse_index_list(row["pos_mcs_atom_indices"])
        n_common = parse_index_list(row["neg_mcs_atom_indices"])
        u_specific = parse_index_list(row["pos_specific_atom_indices"])
        n_specific = parse_index_list(row["neg_specific_atom_indices"])

        validate_indices(
            umol, u_common, u_specific, pair_id, "umami"
        )
        validate_indices(
            nmol, n_common, n_specific, pair_id, "nonumami"
        )

        # Confirm IG atom table also covers the full molecule.
        ig_u_atoms = sorted(udf["atom_index"].astype(int).unique().tolist())
        ig_n_atoms = sorted(ndf["atom_index"].astype(int).unique().tolist())

        if ig_u_atoms != list(range(umol.GetNumAtoms())):
            raise ValueError(
                f"Pair {pair_id}: umami IG atom indices do not cover full molecule"
            )
        if ig_n_atoms != list(range(nmol.GetNumAtoms())):
            raise ValueError(
                f"Pair {pair_id}: non-umami IG atom indices do not cover full molecule"
            )

        u_common_stats = region_stats(udf, u_common)
        u_specific_stats = region_stats(udf, u_specific)
        n_common_stats = region_stats(ndf, n_common)
        n_specific_stats = region_stats(ndf, n_specific)

        u_total_abs = float(udf["abs_ig"].astype(float).sum())
        n_total_abs = float(ndf["abs_ig"].astype(float).sum())

        u_specific_abs_fraction = safe_ratio(
            u_specific_stats["abs_sum"], u_total_abs
        )
        n_specific_abs_fraction = safe_ratio(
            n_specific_stats["abs_sum"], n_total_abs
        )

        u_specific_signed_fraction_of_total_abs = safe_ratio(
            u_specific_stats["signed_sum"], u_total_abs
        )
        n_specific_signed_fraction_of_total_abs = safe_ratio(
            n_specific_stats["signed_sum"], n_total_abs
        )

        rec = {
            "pair_id": pair_id,
            "positive_id": pos_id,
            "negative_id": neg_id,
            "positive_name": pos_name,
            "negative_name": neg_name,
            "similarity": similarity,
            "umami_smiles": usmi,
            "nonumami_smiles": nsmi,
            "mcs_smarts": row["mcs_smarts"],
            "mcs_num_atoms": row["mcs_num_atoms"] if "mcs_num_atoms" in pairs.columns else len(u_common),
            "mcs_num_bonds": row["mcs_num_bonds"] if "mcs_num_bonds" in pairs.columns else np.nan,
            "umami_total_atoms": umol.GetNumAtoms(),
            "nonumami_total_atoms": nmol.GetNumAtoms(),
            "umami_common_atom_indices": ";".join(map(str, u_common)),
            "nonumami_common_atom_indices": ";".join(map(str, n_common)),
            "umami_specific_atom_indices": ";".join(map(str, u_specific)),
            "nonumami_specific_atom_indices": ";".join(map(str, n_specific)),
            "umami_specific_atom_count": len(u_specific),
            "nonumami_specific_atom_count": len(n_specific),
            "umami_total_abs_ig": u_total_abs,
            "nonumami_total_abs_ig": n_total_abs,
            "umami_specific_abs_fraction": u_specific_abs_fraction,
            "nonumami_specific_abs_fraction": n_specific_abs_fraction,
            "specific_abs_fraction_difference": (
                u_specific_abs_fraction - n_specific_abs_fraction
                if not np.isnan(u_specific_abs_fraction)
                and not np.isnan(n_specific_abs_fraction)
                else np.nan
            ),
            "umami_specific_signed_fraction_of_total_abs": (
                u_specific_signed_fraction_of_total_abs
            ),
            "nonumami_specific_signed_fraction_of_total_abs": (
                n_specific_signed_fraction_of_total_abs
            ),
            "specific_signed_sum_difference": (
                u_specific_stats["signed_sum"]
                - n_specific_stats["signed_sum"]
            ),
            "common_signed_sum_difference": (
                u_common_stats["signed_sum"]
                - n_common_stats["signed_sum"]
            ),
            "umami_specific_vs_common_signed_mean_difference": (
                u_specific_stats["signed_mean"]
                - u_common_stats["signed_mean"]
                if not np.isnan(u_specific_stats["signed_mean"])
                and not np.isnan(u_common_stats["signed_mean"])
                else np.nan
            ),
            "nonumami_specific_vs_common_signed_mean_difference": (
                n_specific_stats["signed_mean"]
                - n_common_stats["signed_mean"]
                if not np.isnan(n_specific_stats["signed_mean"])
                and not np.isnan(n_common_stats["signed_mean"])
                else np.nan
            ),
        }

        rec.update(prefix_dict(u_common_stats, "umami_common"))
        rec.update(prefix_dict(u_specific_stats, "umami_specific"))
        rec.update(prefix_dict(n_common_stats, "nonumami_common"))
        rec.update(prefix_dict(n_specific_stats, "nonumami_specific"))

        summary_rows.append(rec)

        # Validation record to demonstrate exact reuse of precomputed MCS partition.
        validation_rows.append({
            "pair_id": pair_id,
            "positive_name": pos_name,
            "negative_name": neg_name,
            "umami_atom_count": umol.GetNumAtoms(),
            "umami_common_count": len(u_common),
            "umami_specific_count": len(u_specific),
            "nonumami_atom_count": nmol.GetNumAtoms(),
            "nonumami_common_count": len(n_common),
            "nonumami_specific_count": len(n_specific),
            "umami_partition_valid": int(
                len(set(u_common) | set(u_specific)) == umol.GetNumAtoms()
                and len(set(u_common) & set(u_specific)) == 0
            ),
            "nonumami_partition_valid": int(
                len(set(n_common) | set(n_specific)) == nmol.GetNumAtoms()
                and len(set(n_common) & set(n_specific)) == 0
            ),
        })

        u_common_set = set(u_common)
        n_common_set = set(n_common)

        for _, ar in udf.iterrows():
            atom_idx = int(ar["atom_index"])
            rr = ar.to_dict()
            atom = umol.GetAtomWithIdx(atom_idx)
            rr.update({
                "pair_id": pair_id,
                "positive_id": pos_id,
                "negative_id": neg_id,
                "positive_name": pos_name,
                "negative_name": neg_name,
                "pair_role": "umami",
                "pair_similarity": similarity,
                "pair_other_smiles": nsmi,
                "mcs_region": (
                    "common"
                    if atom_idx in u_common_set
                    else "umami_specific"
                ),
                "mcs_smarts": row["mcs_smarts"],
                "atom_symbol_rdkit": atom.GetSymbol(),
            })
            atom_rows.append(rr)

        for _, ar in ndf.iterrows():
            atom_idx = int(ar["atom_index"])
            rr = ar.to_dict()
            atom = nmol.GetAtomWithIdx(atom_idx)
            rr.update({
                "pair_id": pair_id,
                "positive_id": pos_id,
                "negative_id": neg_id,
                "positive_name": pos_name,
                "negative_name": neg_name,
                "pair_role": "nonumami",
                "pair_similarity": similarity,
                "pair_other_smiles": usmi,
                "mcs_region": (
                    "common"
                    if atom_idx in n_common_set
                    else "nonumami_specific"
                ),
                "mcs_smarts": row["mcs_smarts"],
                "atom_symbol_rdkit": atom.GetSymbol(),
            })
            atom_rows.append(rr)

    summary_df = pd.DataFrame(summary_rows)
    atom_annot_df = pd.DataFrame(atom_rows)
    validation_df = pd.DataFrame(validation_rows)

    summary_df.to_csv(
        outdir / "matched_pair_ig_summary.csv", index=False
    )
    atom_annot_df.to_csv(
        outdir / "matched_pair_atom_annotations.csv", index=False
    )
    validation_df.to_csv(
        outdir / "matched_pair_partition_validation.csv", index=False
    )

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(
            outdir / "matched_pair_missing_from_ig.csv", index=False
        )

    stats_rows = []

    def paired_test(name, col_a, col_b):
        sub = summary_df[[col_a, col_b]].dropna()
        if len(sub) == 0:
            return

        x = sub[col_a].astype(float).values
        y = sub[col_b].astype(float).values
        diff = x - y

        if np.allclose(diff, 0):
            stat, pval = 0.0, 1.0
        else:
            try:
                stat, pval = wilcoxon(
                    x, y, alternative="two-sided", zero_method="wilcox"
                )
            except ValueError:
                stat, pval = np.nan, np.nan

        stats_rows.append({
            "comparison": name,
            "n_pairs": len(sub),
            "median_a": float(np.median(x)),
            "median_b": float(np.median(y)),
            "median_difference_a_minus_b": float(np.median(diff)),
            "mean_difference_a_minus_b": float(np.mean(diff)),
            "fraction_a_greater_b": float(np.mean(x > y)),
            "wilcoxon_stat": stat,
            "wilcoxon_p": pval,
        })

    if len(summary_df):
        paired_test(
            "umami_specific_signed_sum_vs_nonumami_specific_signed_sum",
            "umami_specific_signed_sum",
            "nonumami_specific_signed_sum",
        )
        paired_test(
            "umami_specific_signed_mean_vs_umami_common_signed_mean",
            "umami_specific_signed_mean",
            "umami_common_signed_mean",
        )
        paired_test(
            "umami_specific_abs_fraction_vs_nonumami_specific_abs_fraction",
            "umami_specific_abs_fraction",
            "nonumami_specific_abs_fraction",
        )
        paired_test(
            "umami_common_signed_sum_vs_nonumami_common_signed_sum",
            "umami_common_signed_sum",
            "nonumami_common_signed_sum",
        )
        paired_test(
            "umami_specific_signed_fraction_vs_nonumami_specific_signed_fraction",
            "umami_specific_signed_fraction_of_total_abs",
            "nonumami_specific_signed_fraction_of_total_abs",
        )

    stats_df = pd.DataFrame(stats_rows)

    if len(stats_df):
        mask = stats_df["wilcoxon_p"].notna()
        stats_df["wilcoxon_q"] = np.nan
        if mask.any():
            stats_df.loc[mask, "wilcoxon_q"] = bh_fdr(
                stats_df.loc[mask, "wilcoxon_p"].values
            )

    stats_df.to_csv(
        outdir / "matched_pair_group_stats.csv", index=False
    )

    analysis_summary = pd.DataFrame([
        {"item": "input_pairs", "value": len(pairs)},
        {"item": "successfully_analyzed_pairs", "value": len(summary_df)},
        {"item": "pairs_missing_from_ig", "value": len(missing_rows)},
        {
            "item": "all_partitions_valid",
            "value": int(
                len(validation_df) > 0
                and validation_df["umami_partition_valid"].all()
                and validation_df["nonumami_partition_valid"].all()
            ),
        },
        {
            "item": "median_mcs_atoms",
            "value": (
                float(summary_df["mcs_num_atoms"].median())
                if len(summary_df) else np.nan
            ),
        },
        {
            "item": "median_umami_specific_atoms",
            "value": (
                float(summary_df["umami_specific_atom_count"].median())
                if len(summary_df) else np.nan
            ),
        },
        {
            "item": "median_nonumami_specific_atoms",
            "value": (
                float(summary_df["nonumami_specific_atom_count"].median())
                if len(summary_df) else np.nan
            ),
        },
    ])

    analysis_summary.to_csv(
        outdir / "matched_pair_analysis_summary.csv", index=False
    )

    print("Done.")
    print(f"Input pairs: {len(pairs)}")
    print(f"Successfully analyzed pairs: {len(summary_df)}")
    print(f"Missing from IG: {len(missing_rows)}")

    if len(validation_df):
        all_valid = (
            validation_df["umami_partition_valid"].all()
            and validation_df["nonumami_partition_valid"].all()
        )
        print(f"All precomputed partitions valid: {all_valid}")

    if len(summary_df):
        print(
            f"Median MCS atoms: "
            f"{summary_df['mcs_num_atoms'].median():.1f}"
        )
        print(
            "Median umami-specific atoms: "
            f"{summary_df['umami_specific_atom_count'].median():.1f}"
        )
        print(
            "Median non-umami-specific atoms: "
            f"{summary_df['nonumami_specific_atom_count'].median():.1f}"
        )

    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()

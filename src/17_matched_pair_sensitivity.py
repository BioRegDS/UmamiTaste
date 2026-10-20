#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


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
        prev = min(prev, ranked[i] * n / rank)
        adj[i] = min(prev, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def paired_wilcoxon(df, col_a, col_b):
    sub = df[[col_a, col_b]].dropna()
    if len(sub) == 0:
        return dict(n_pairs=0, median_a=np.nan, median_b=np.nan,
                    median_difference=np.nan, mean_difference=np.nan,
                    fraction_a_greater_b=np.nan, wilcoxon_stat=np.nan,
                    p_value=np.nan)
    a = sub[col_a].astype(float).to_numpy()
    b = sub[col_b].astype(float).to_numpy()
    diff = a - b
    if np.allclose(diff, 0):
        stat, pval = 0.0, 1.0
    else:
        try:
            stat, pval = wilcoxon(a, b, alternative="two-sided",
                                  zero_method="wilcox")
        except ValueError:
            stat, pval = np.nan, np.nan
    return dict(
        n_pairs=len(sub),
        median_a=float(np.median(a)),
        median_b=float(np.median(b)),
        median_difference=float(np.median(diff)),
        mean_difference=float(np.mean(diff)),
        fraction_a_greater_b=float(np.mean(a > b)),
        wilcoxon_stat=stat,
        p_value=pval,
    )


def get_fold_assignments(atom_df):
    rows = []
    for (pair_id, role), g in atom_df.groupby(["pair_id", "pair_role"]):
        folds = sorted(pd.unique(g["fold"].dropna()))
        if len(folds) != 1:
            raise ValueError(f"Pair {pair_id}, role {role}: folds={folds}")
        rows.append({"pair_id": pair_id, "pair_role": role, "fold": int(folds[0])})
    wide = pd.DataFrame(rows).pivot(index="pair_id", columns="pair_role",
                                    values="fold").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"umami": "umami_fold",
                                "nonumami": "nonumami_fold"})
    if "umami_fold" not in wide or "nonumami_fold" not in wide:
        raise ValueError(f"Could not recover both folds: {list(wide.columns)}")
    wide["same_fold"] = (wide["umami_fold"] == wide["nonumami_fold"]).astype(int)
    return wide


def greedy_unique_pairs(df):
    work = df.sort_values(["similarity", "pair_id"],
                          ascending=[False, True], kind="mergesort").copy()
    used = set()
    keep = []
    for _, row in work.iterrows():
        u = str(row["umami_smiles"])
        n = str(row["nonumami_smiles"])
        if u in used or n in used:
            continue
        keep.append(row["pair_id"])
        used.add(u)
        used.add(n)
    return keep


def duplicate_usage(df):
    counts = Counter()
    for _, row in df.iterrows():
        counts[("umami", str(row["umami_smiles"]))] += 1
        counts[("nonumami", str(row["nonumami_smiles"]))] += 1
    rows = [{"role": role, "smiles": smi, "pair_count": count}
            for (role, smi), count in counts.items()]
    return pd.DataFrame(rows).sort_values(
        ["pair_count", "role", "smiles"],
        ascending=[False, True, True]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary",
                   default="data/ig/matched_pairs/matched_pair_ig_summary.csv")
    p.add_argument("--atoms",
                   default="data/ig/matched_pairs/matched_pair_atom_annotations.csv")
    p.add_argument("--output-dir",
                   default="data/ig/matched_pairs/sensitivity")
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    atoms = pd.read_csv(args.atoms)

    dup = duplicate_usage(summary)
    dup.to_csv(outdir / "matched_pair_duplicate_usage.csv", index=False)

    folds = get_fold_assignments(atoms)
    folds.to_csv(outdir / "matched_pair_fold_assignments.csv", index=False)

    summary = summary.merge(folds, on="pair_id", how="left", validate="one_to_one")
    if summary["same_fold"].isna().any():
        raise ValueError("Missing fold assignment after merge.")

    unique_ids = greedy_unique_pairs(summary)
    summary["unique_compound_pair"] = summary["pair_id"].isin(unique_ids).astype(int)
    summary["unique_and_same_fold"] = (
        (summary["unique_compound_pair"] == 1) &
        (summary["same_fold"] == 1)
    ).astype(int)

    summary[[
        "pair_id", "positive_name", "negative_name", "similarity",
        "umami_fold", "nonumami_fold", "same_fold",
        "unique_compound_pair", "unique_and_same_fold"
    ]].to_csv(outdir / "matched_pair_sensitivity_subsets.csv", index=False)

    metrics = [
        ("specific_signed_sum",
         "umami_specific_signed_sum",
         "nonumami_specific_signed_sum"),
        ("specific_signed_mean_per_atom",
         "umami_specific_signed_mean",
         "nonumami_specific_signed_mean"),
        ("specific_signed_fraction_total_abs",
         "umami_specific_signed_fraction_of_total_abs",
         "nonumami_specific_signed_fraction_of_total_abs"),
        ("specific_abs_fraction",
         "umami_specific_abs_fraction",
         "nonumami_specific_abs_fraction"),
    ]

    subsets = [
        ("all_pairs", summary),
        ("unique_compounds", summary[summary["unique_compound_pair"] == 1]),
        ("same_fold", summary[summary["same_fold"] == 1]),
        ("unique_and_same_fold", summary[summary["unique_and_same_fold"] == 1]),
    ]

    rows = []
    for subset_name, subdf in subsets:
        for metric_name, col_a, col_b in metrics:
            res = paired_wilcoxon(subdf, col_a, col_b)
            res.update({
                "subset": subset_name,
                "metric": metric_name,
                "column_umami": col_a,
                "column_nonumami": col_b,
            })
            rows.append(res)

    results = pd.DataFrame(rows)
    valid = results["p_value"].notna()
    results["fdr_q"] = np.nan
    if valid.any():
        results.loc[valid, "fdr_q"] = bh_fdr(
            results.loc[valid, "p_value"].to_numpy()
        )
    results.to_csv(outdir / "matched_pair_sensitivity_results.csv", index=False)

    repeated = dup[dup["pair_count"] > 1]
    lines = [
        "Matched-pair sensitivity analysis",
        "=" * 50,
        f"All pairs: {len(summary)}",
        f"Unique-compound greedy subset: {int(summary['unique_compound_pair'].sum())}",
        f"Same-fold subset: {int(summary['same_fold'].sum())}",
        f"Unique + same-fold subset: {int(summary['unique_and_same_fold'].sum())}",
        f"Compounds appearing in >1 pair: {len(repeated)}",
        f"Maximum pair reuse count for one compound: {int(dup['pair_count'].max()) if len(dup) else 0}",
        "",
    ]

    for subset_name, _ in subsets:
        lines.append(f"[{subset_name}]")
        tmp = results[results["subset"] == subset_name]
        for _, r in tmp.iterrows():
            lines.append(
                f"  {r['metric']}: n={int(r['n_pairs'])}, "
                f"median umami={r['median_a']:.4g}, "
                f"median non-umami={r['median_b']:.4g}, "
                f"umami>non-umami={r['fraction_a_greater_b']:.3f}, "
                f"p={r['p_value']:.4g}, q={r['fdr_q']:.4g}"
            )
        lines.append("")

    (outdir / "matched_pair_sensitivity_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()

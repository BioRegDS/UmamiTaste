#!/usr/bin/env python3
"""
Prepare scaffold-disjoint K-fold cross-validation splits for kMoL.

The script groups molecules by Bemis-Murcko scaffold and assigns whole
scaffold groups to folds while approximately balancing total size and class
counts. It writes a kMoL PrecomputedSplitter JSON and audit CSV files.

Example:
  python src/07_prepare_scaffold_cv.py \
    --input data/kmol/umami_strict.csv \
    --output-dir data/kmol/scaffold_cv \
    --folds 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="kMoL CSV containing smiles and binary target")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--label-col", default="umami")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    # Match kMoL ScaffoldDividerSplitter behavior: chirality is not included.
    return MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def objective(fold_stats, target_n, target_pos, target_neg):
    # Normalize each component so that size and class balance have comparable weight.
    score = 0.0
    for s in fold_stats:
        score += ((s["n"] - target_n) / max(target_n, 1.0)) ** 2
        score += ((s["pos"] - target_pos) / max(target_pos, 1.0)) ** 2
        score += ((s["neg"] - target_neg) / max(target_neg, 1.0)) ** 2
    return score


def assign_groups(groups, k: int, seed: int):
    rng = random.Random(seed)
    total_n = sum(g["n"] for g in groups)
    total_pos = sum(g["pos"] for g in groups)
    total_neg = sum(g["neg"] for g in groups)
    target_n = total_n / k
    target_pos = total_pos / k
    target_neg = total_neg / k

    # Large and class-extreme groups first; random tie-breaker makes seed explicit.
    work = []
    for g in groups:
        gg = dict(g)
        gg["tie"] = rng.random()
        work.append(gg)
    work.sort(key=lambda g: (-g["n"], -abs(g["pos"] - g["neg"]), g["tie"]))

    folds = [[] for _ in range(k)]
    stats = [{"n": 0, "pos": 0, "neg": 0} for _ in range(k)]

    for g in work:
        candidates = []
        for f in range(k):
            trial = [dict(x) for x in stats]
            trial[f]["n"] += g["n"]
            trial[f]["pos"] += g["pos"]
            trial[f]["neg"] += g["neg"]
            candidates.append((objective(trial, target_n, target_pos, target_neg), stats[f]["n"], rng.random(), f))
        _, _, _, best = min(candidates)
        folds[best].append(g)
        stats[best]["n"] += g["n"]
        stats[best]["pos"] += g["pos"]
        stats[best]["neg"] += g["neg"]

    # Simple local improvement: move one whole scaffold group if objective improves.
    improved = True
    while improved:
        improved = False
        base = objective(stats, target_n, target_pos, target_neg)
        best_move = None
        best_score = base
        for a in range(k):
            if len(folds[a]) <= 1:
                continue
            for gi, g in enumerate(folds[a]):
                for b in range(k):
                    if a == b:
                        continue
                    trial = [dict(x) for x in stats]
                    for key in ("n", "pos", "neg"):
                        trial[a][key] -= g[key]
                        trial[b][key] += g[key]
                    sc = objective(trial, target_n, target_pos, target_neg)
                    if sc + 1e-12 < best_score:
                        best_score = sc
                        best_move = (a, gi, b, trial)
        if best_move is not None:
            a, gi, b, trial = best_move
            g = folds[a].pop(gi)
            folds[b].append(g)
            stats = trial
            improved = True

    return folds, stats


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    missing = [c for c in (args.smiles_col, args.label_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}; columns={list(df.columns)}")

    labels = pd.to_numeric(df[args.label_col], errors="raise").astype(float)
    if not set(labels.unique()).issubset({0.0, 1.0}):
        raise ValueError(f"{args.label_col} must contain only 0/1; observed {sorted(labels.unique())}")

    df = df.copy()
    df["row_id"] = np.arange(len(df), dtype=int)
    df["murcko_scaffold"] = [scaffold_from_smiles(s) for s in df[args.smiles_col]]
    df["is_acyclic_scaffold"] = df["murcko_scaffold"].eq("")

    groups = []
    for scaffold, g in df.groupby("murcko_scaffold", sort=False, dropna=False):
        pos = int(g[args.label_col].astype(float).sum())
        groups.append({
            "scaffold": scaffold,
            "ids": g["row_id"].astype(int).tolist(),
            "n": int(len(g)),
            "pos": pos,
            "neg": int(len(g) - pos),
        })

    folds, stats = assign_groups(groups, args.folds, args.seed)

    # Ensure every row is assigned exactly once and scaffold leakage is impossible.
    split_dict = {}
    assignment_rows = []
    for f, glist in enumerate(folds):
        ids = sorted([i for g in glist for i in g["ids"]])
        split_dict[f"fold_{f}"] = ids
        for g in glist:
            for i in g["ids"]:
                assignment_rows.append({"row_id": i, "fold": f, "murcko_scaffold": g["scaffold"]})

    all_ids = sorted(i for ids in split_dict.values() for i in ids)
    expected = list(range(len(df)))
    if all_ids != expected:
        raise RuntimeError("Fold assignment does not cover each input row exactly once")

    assign = pd.DataFrame(assignment_rows).sort_values("row_id")
    check = assign.groupby("murcko_scaffold")["fold"].nunique()
    leaking = check[check > 1]
    if len(leaking):
        raise RuntimeError(f"Scaffold leakage detected for {len(leaking)} scaffolds")

    # Audit table with molecule details.
    audit = df.merge(assign[["row_id", "fold"]], on="row_id", how="left")
    audit.to_csv(outdir / "scaffold_fold_assignments.csv", index=False)

    scaffold_rows = []
    for f, glist in enumerate(folds):
        for g in glist:
            scaffold_rows.append({
                "fold": f,
                "murcko_scaffold": g["scaffold"],
                "n": g["n"],
                "positive": g["pos"],
                "negative": g["neg"],
                "acyclic_scaffold": g["scaffold"] == "",
            })
    pd.DataFrame(scaffold_rows).sort_values(["fold", "n"], ascending=[True, False]).to_csv(
        outdir / "scaffold_groups_by_fold.csv", index=False
    )

    summary = []
    for f in range(args.folds):
        sub = audit[audit["fold"] == f]
        n = len(sub)
        pos = int(sub[args.label_col].astype(float).sum())
        summary.append({
            "fold": f,
            "n": n,
            "positive": pos,
            "negative": n - pos,
            "positive_fraction": pos / n if n else np.nan,
            "unique_scaffolds": int(sub["murcko_scaffold"].nunique()),
            "acyclic_molecules": int(sub["is_acyclic_scaffold"].sum()),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(outdir / "scaffold_fold_summary.csv", index=False)

    with open(outdir / "scaffold_5fold_splits.json", "w") as f:
        json.dump(split_dict, f, indent=2)

    print("\nScaffold-disjoint CV split created")
    print(f"Input records: {len(df)}")
    print(f"Unique Murcko scaffolds: {df['murcko_scaffold'].nunique()}")
    print(f"Acyclic/empty-scaffold molecules: {int(df['is_acyclic_scaffold'].sum())}")
    print("\nFold summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {outdir / 'scaffold_5fold_splits.json'}")
    print("Scaffold leakage check: PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare model-ready umami training sets and chemical-space diagnostics.

Inputs
------
- Final umami-positive dataset from 02_finalize_umami_dataset.py
- Primary negative dataset from 03_prepare_negative_dataset.py
- Strict (tasteless-only) negative dataset from 03_prepare_negative_dataset.py

Outputs
-------
1) Full primary dataset: all positives + all primary negatives
2) Strict dataset: all positives + all tasteless negatives
3) Repeated random-balanced datasets: n_pos positives + n_pos sampled primary negatives
4) Physicochemical-matched balanced dataset: one unique primary negative matched to each positive
5) Chemical-space summaries and matching diagnostics
6) Optional PCA and descriptor-distribution plots

Matching strategy
-----------------
The matched balanced set uses standardized physicochemical descriptors:
MW, MolLogP, TPSA, HBD, HBA, and rotatable bonds.
Each positive is greedily assigned a unique negative with the smallest Euclidean
 distance in z-scored descriptor space. Structural similarity (Morgan/ECFP4
Tanimoto) is reported for diagnostics but is NOT used to choose the matched
negative, so the matching does not deliberately select near-identical structural
analogs.

Example
-------
python src/04_prepare_training_sets.py \
  --positive data/processed/umami_positive_final.csv \
  --negative-primary data/processed/negative_primary_final.csv \
  --negative-strict data/processed/negative_strict_final.csv \
  --output-dir data \
  --seed 42 \
  --random-repeats 10
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DESC_COLS = ["calc_MW", "calc_LogP", "calc_TPSA", "calc_HBD", "calc_HBA", "calc_RotB"]


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def murcko_scaffold(mol: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return ""
        return canonical_smiles(scaf)
    except Exception:
        return ""


def calc_descriptors(smiles: str) -> Dict[str, float | int | str]:
    mol = mol_from_smiles(smiles)
    return {
        "calc_MW": float(Descriptors.MolWt(mol)),
        "calc_LogP": float(Crippen.MolLogP(mol)),
        "calc_TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "calc_HBD": int(Lipinski.NumHDonors(mol)),
        "calc_HBA": int(Lipinski.NumHAcceptors(mol)),
        "calc_RotB": int(Lipinski.NumRotatableBonds(mol)),
        "calc_HeavyAtoms": int(mol.GetNumHeavyAtoms()),
        "calc_Scaffold": murcko_scaffold(mol),
    }


def normalize_input(df: pd.DataFrame, label: int, source_set: str) -> pd.DataFrame:
    out = df.copy()
    if "ml_SMILES" not in out.columns:
        if "parent_SMILES" in out.columns:
            out["ml_SMILES"] = out["parent_SMILES"]
        else:
            raise ValueError(f"{source_set}: input needs ml_SMILES or parent_SMILES")

    out["ml_SMILES"] = out["ml_SMILES"].astype(str).str.strip()
    if out["ml_SMILES"].eq("").any():
        raise ValueError(f"{source_set}: empty ml_SMILES found")

    out["label"] = int(label)
    out["label_name"] = "umami" if label == 1 else "non_umami"
    out["source_set"] = source_set

    desc = pd.DataFrame([calc_descriptors(s) for s in out["ml_SMILES"]], index=out.index)
    for c in desc.columns:
        out[c] = desc[c]

    if out["ml_SMILES"].duplicated().any():
        dups = out.loc[out["ml_SMILES"].duplicated(False), "ml_SMILES"].unique()
        raise ValueError(f"{source_set}: duplicate ml_SMILES remain ({len(dups)} structures)")
    return out


def check_cross_conflicts(pos: pd.DataFrame, neg: pd.DataFrame, tag: str) -> None:
    common = set(pos["ml_SMILES"]) & set(neg["ml_SMILES"])
    if common:
        raise ValueError(f"Positive/negative structure conflict remains in {tag}: {len(common)} structures")


def compact_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "ml_SMILES", "label", "label_name", "source_set", "training_set",
        "Name", "representative_ID", "ID", "merged_record_count", "merged_IDs",
        "merged_Names", "merged_negative_classes", "Reference", "merged_References",
        "calc_MW", "calc_LogP", "calc_TPSA", "calc_HBD", "calc_HBA", "calc_RotB",
        "calc_HeavyAtoms", "calc_Scaffold", "parent_SMILES", "parent_MurckoScaffold",
    ]
    cols = [c for c in preferred if c in df.columns]
    return df.reindex(columns=cols)


def combine(pos: pd.DataFrame, neg: pd.DataFrame, name: str, seed: int | None = None) -> pd.DataFrame:
    p = pos.copy()
    n = neg.copy()
    p["training_set"] = name
    n["training_set"] = name
    out = pd.concat([p, n], ignore_index=True)
    if seed is not None:
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return compact_model_frame(out)


def class_weights(df: pd.DataFrame) -> Dict[int, float]:
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    n_classes = len(counts)
    return {int(k): total / (n_classes * v) for k, v in counts.items()}


def dataset_summary(name: str, df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, label_name in [(1, "umami"), (0, "non_umami")]:
        g = df[df["label"] == label]
        if len(g) == 0:
            continue
        scaffold_values = g["calc_Scaffold"].fillna("").astype(str)
        nonempty_scaf = scaffold_values[scaffold_values != ""]
        row = {
            "dataset": name,
            "label": label,
            "label_name": label_name,
            "n": len(g),
            "unique_scaffolds_nonempty": int(nonempty_scaf.nunique()),
            "scaffold_fraction_unique": float(nonempty_scaf.nunique() / len(g)) if len(g) else np.nan,
        }
        for c in DESC_COLS:
            row[f"{c}_median"] = float(g[c].median())
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_q25"] = float(g[c].quantile(0.25))
            row[f"{c}_q75"] = float(g[c].quantile(0.75))
        rows.append(row)
    return rows


def standardized_descriptor_matrix(pos: pd.DataFrame, neg: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    pooled = pd.concat([pos[DESC_COLS], neg[DESC_COLS]], ignore_index=True).astype(float)
    scaler = StandardScaler().fit(pooled)
    return scaler.transform(pos[DESC_COLS]), scaler.transform(neg[DESC_COLS]), scaler


def greedy_physchem_match(pos: pd.DataFrame, neg: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match one unique negative to each positive by descriptor distance."""
    if len(neg) < len(pos):
        raise ValueError("Need at least as many negatives as positives for 1:1 matching")

    zp, zn, _ = standardized_descriptor_matrix(pos, neg)
    # Pairwise squared Euclidean distance: shape n_pos x n_neg
    d2 = ((zp[:, None, :] - zn[None, :, :]) ** 2).sum(axis=2)

    # Match the positives with the fewest close alternatives first. This is a
    # deterministic greedy approximation that reduces late poor matches.
    sorted_dist = np.sort(d2, axis=1)
    k = min(10, d2.shape[1])
    difficulty = sorted_dist[:, :k].mean(axis=1)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0, 1e-10, size=len(pos))
    order = np.argsort(-(difficulty + jitter))  # hardest first

    available = set(range(len(neg)))
    assignments: List[Tuple[int, int, float]] = []
    for pi in order:
        candidates = np.fromiter(available, dtype=int)
        local = d2[pi, candidates]
        best_local = int(np.argmin(local))
        ni = int(candidates[best_local])
        assignments.append((int(pi), ni, float(math.sqrt(d2[pi, ni]))))
        available.remove(ni)

    assignments.sort(key=lambda x: x[0])
    neg_indices = [ni for _, ni, _ in assignments]
    matched = neg.iloc[neg_indices].copy().reset_index(drop=True)

    diag_rows = []
    for match_id, (pi, ni, dist) in enumerate(assignments, start=1):
        prow = pos.iloc[pi]
        nrow = neg.iloc[ni]
        diag_rows.append({
            "match_id": match_id,
            "positive_index": int(pi),
            "negative_index": int(ni),
            "positive_name": prow.get("Name", ""),
            "negative_name": nrow.get("Name", ""),
            "positive_SMILES": prow["ml_SMILES"],
            "negative_SMILES": nrow["ml_SMILES"],
            "descriptor_distance_z": dist,
        })
    return matched, pd.DataFrame(diag_rows)


def morgan_fps(smiles: Iterable[str]):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for s in smiles:
        fps.append(gen.GetFingerprint(mol_from_smiles(s)))
    return fps


def add_tanimoto_to_match_diagnostics(diag: pd.DataFrame) -> pd.DataFrame:
    out = diag.copy()
    vals = []
    for _, r in out.iterrows():
        fps = morgan_fps([r["positive_SMILES"], r["negative_SMILES"]])
        vals.append(float(DataStructs.TanimotoSimilarity(fps[0], fps[1])))
    out["matched_ECFP4_tanimoto"] = vals
    return out


def nearest_negative_similarity(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    """For every positive, report the most structurally similar negative in the full primary pool."""
    p_fps = morgan_fps(pos["ml_SMILES"])
    n_fps = morgan_fps(neg["ml_SMILES"])
    rows = []
    for i, fp in enumerate(p_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, n_fps), dtype=float)
        j = int(np.argmax(sims))
        rows.append({
            "positive_index": i,
            "positive_name": pos.iloc[i].get("Name", ""),
            "positive_SMILES": pos.iloc[i]["ml_SMILES"],
            "nearest_negative_index": j,
            "nearest_negative_name": neg.iloc[j].get("Name", ""),
            "nearest_negative_SMILES": neg.iloc[j]["ml_SMILES"],
            "max_ECFP4_tanimoto": float(sims[j]),
        })
    return pd.DataFrame(rows)


def pca_coordinates(pos: pd.DataFrame, neg: pd.DataFrame, seed: int) -> pd.DataFrame:
    """2D PCA of ECFP4 fingerprints for a reproducible chemical-space diagnostic."""
    all_df = pd.concat([pos.assign(_group="umami"), neg.assign(_group="primary_negative")], ignore_index=True)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    arr = np.zeros((len(all_df), 2048), dtype=np.float32)
    for i, s in enumerate(all_df["ml_SMILES"]):
        fp = gen.GetFingerprint(mol_from_smiles(s))
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    pca = PCA(n_components=2, random_state=seed)
    xy = pca.fit_transform(arr)
    out = pd.DataFrame({
        "ml_SMILES": all_df["ml_SMILES"].values,
        "label": all_df["label"].values,
        "group": all_df["_group"].values,
        "PC1": xy[:, 0],
        "PC2": xy[:, 1],
    })
    out.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    return out


def make_plots(pca_df: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: Path) -> None:
    """Optional light-weight plots; uses matplotlib defaults intentionally."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting skipped: matplotlib unavailable ({exc})")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label_name, marker in [("umami", "o"), ("primary_negative", ".")]:
        g = pca_df[pca_df["group"] == label_name]
        ax.scatter(g["PC1"], g["PC2"], s=28 if label_name == "umami" else 10,
                   alpha=0.8 if label_name == "umami" else 0.35, label=label_name, marker=marker)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("ECFP4 chemical space: umami vs primary negatives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "chemical_space_pca.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare umami model training sets and chemical-space diagnostics")
    parser.add_argument("--positive", required=True, help="umami_positive_final.csv")
    parser.add_argument("--negative-primary", required=True, help="negative_primary_final.csv")
    parser.add_argument("--negative-strict", required=True, help="negative_strict_final.csv")
    parser.add_argument("--output-dir", default="data", help="Project data directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-repeats", type=int, default=10, help="Number of independent 1:1 random-balanced sets")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG diagnostic plots")
    args = parser.parse_args()

    if args.random_repeats < 1:
        raise ValueError("--random-repeats must be >= 1")

    outdir = Path(args.output_dir)
    model_dir = outdir / "model_sets"
    analysis_dir = outdir / "analysis"
    figures_dir = outdir / "figures"
    for p in [model_dir, analysis_dir, figures_dir]:
        p.mkdir(parents=True, exist_ok=True)

    pos = normalize_input(pd.read_csv(args.positive), 1, "positive")
    neg_primary = normalize_input(pd.read_csv(args.negative_primary), 0, "primary_negative")
    neg_strict = normalize_input(pd.read_csv(args.negative_strict), 0, "strict_negative")

    check_cross_conflicts(pos, neg_primary, "primary")
    check_cross_conflicts(pos, neg_strict, "strict")

    print("=== Input counts ===")
    print(f"Positive: {len(pos)}")
    print(f"Primary negatives: {len(neg_primary)}")
    print(f"Strict negatives: {len(neg_strict)}")

    # 1) Full primary
    full_primary = combine(pos, neg_primary, "full_primary", seed=args.seed)
    full_primary.to_csv(model_dir / "training_full_primary.csv", index=False)

    # 2) Strict
    strict = combine(pos, neg_strict, "strict_tasteless", seed=args.seed)
    strict.to_csv(model_dir / "training_strict_tasteless.csv", index=False)

    # 3) Repeated random 1:1 balanced sets
    random_manifest = []
    for r in range(1, args.random_repeats + 1):
        rseed = args.seed + r - 1
        sample = neg_primary.sample(n=len(pos), replace=False, random_state=rseed).copy()
        ds = combine(pos, sample, f"balanced_random_r{r:02d}", seed=rseed)
        path = model_dir / f"training_balanced_random_r{r:02d}.csv"
        ds.to_csv(path, index=False)
        random_manifest.append({
            "repeat": r,
            "seed": rseed,
            "n_positive": len(pos),
            "n_negative": len(sample),
            "file": str(path),
        })
    pd.DataFrame(random_manifest).to_csv(analysis_dir / "balanced_random_manifest.csv", index=False)

    # 4) Physicochemical-matched 1:1 balanced set
    matched_neg, match_diag = greedy_physchem_match(pos, neg_primary, args.seed)
    match_diag = add_tanimoto_to_match_diagnostics(match_diag)
    matched_ds = combine(pos, matched_neg, "balanced_physchem_matched", seed=args.seed)
    matched_ds.to_csv(model_dir / "training_balanced_physchem_matched.csv", index=False)
    match_diag.to_csv(analysis_dir / "physchem_matched_pairs.csv", index=False)

    # Nearest structural negative diagnostic across the full primary pool
    nearest = nearest_negative_similarity(pos, neg_primary)
    nearest.to_csv(analysis_dir / "positive_nearest_primary_negative_ecfp.csv", index=False)

    # 5) Summaries and class weights
    summary_rows: List[Dict[str, object]] = []
    datasets_for_summary = {
        "full_primary": full_primary,
        "strict_tasteless": strict,
        "balanced_physchem_matched": matched_ds,
    }
    # Include the first random balanced set as a representative summary.
    random1 = pd.read_csv(model_dir / "training_balanced_random_r01.csv")
    datasets_for_summary["balanced_random_r01"] = random1

    weight_rows = []
    for name, ds in datasets_for_summary.items():
        summary_rows.extend(dataset_summary(name, ds))
        weights = class_weights(ds)
        for label, w in sorted(weights.items()):
            weight_rows.append({
                "dataset": name,
                "label": label,
                "label_name": "umami" if label == 1 else "non_umami",
                "class_weight": w,
                "n": int((ds["label"] == label).sum()),
            })
    pd.DataFrame(summary_rows).to_csv(analysis_dir / "chemical_space_summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(analysis_dir / "recommended_class_weights.csv", index=False)

    # 6) PCA diagnostic on full primary chemical space
    pca_df = pca_coordinates(pos, neg_primary, args.seed)
    pca_df.to_csv(analysis_dir / "chemical_space_pca_coordinates.csv", index=False)
    # Save PCA variance explicitly because DataFrame attrs are not preserved in CSV.
    # Recompute cheaply from coordinate source to retain a simple audit file.
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    all_smi = list(pos["ml_SMILES"]) + list(neg_primary["ml_SMILES"])
    arr = np.zeros((len(all_smi), 2048), dtype=np.float32)
    for i, s in enumerate(all_smi):
        fp = gen.GetFingerprint(mol_from_smiles(s))
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    pca_check = PCA(n_components=2, random_state=args.seed).fit(arr)
    pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca_check.explained_variance_ratio_,
    }).to_csv(analysis_dir / "chemical_space_pca_variance.csv", index=False)

    if not args.no_plots:
        make_plots(pca_df, pd.DataFrame(summary_rows), figures_dir)

    # Global run summary
    run_summary = pd.DataFrame([
        {"metric": "positive_n", "value": len(pos)},
        {"metric": "primary_negative_n", "value": len(neg_primary)},
        {"metric": "strict_negative_n", "value": len(neg_strict)},
        {"metric": "full_primary_total_n", "value": len(full_primary)},
        {"metric": "strict_total_n", "value": len(strict)},
        {"metric": "balanced_each_class_n", "value": len(pos)},
        {"metric": "random_balanced_repeats", "value": args.random_repeats},
        {"metric": "matched_pair_median_descriptor_distance_z", "value": float(match_diag["descriptor_distance_z"].median())},
        {"metric": "matched_pair_median_ECFP4_tanimoto", "value": float(match_diag["matched_ECFP4_tanimoto"].median())},
        {"metric": "positive_nearest_negative_median_max_ECFP4_tanimoto", "value": float(nearest["max_ECFP4_tanimoto"].median())},
        {"metric": "positive_nearest_negative_max_ECFP4_tanimoto", "value": float(nearest["max_ECFP4_tanimoto"].max())},
    ])
    run_summary.to_csv(analysis_dir / "training_set_summary.csv", index=False)

    print("\n=== Model-ready datasets ===")
    print(model_dir / "training_full_primary.csv")
    print(model_dir / "training_strict_tasteless.csv")
    print(model_dir / "training_balanced_physchem_matched.csv")
    print(model_dir / "training_balanced_random_r01.csv", f"... r{args.random_repeats:02d}")

    print("\n=== Diagnostics ===")
    print(analysis_dir / "chemical_space_summary.csv")
    print(analysis_dir / "recommended_class_weights.csv")
    print(analysis_dir / "physchem_matched_pairs.csv")
    print(analysis_dir / "positive_nearest_primary_negative_ecfp.csv")
    print(analysis_dir / "chemical_space_pca_coordinates.csv")

    print("\n=== Key numbers ===")
    print(f"Full primary: {int((full_primary.label==1).sum())} positive / {int((full_primary.label==0).sum())} negative")
    print(f"Strict: {int((strict.label==1).sum())} positive / {int((strict.label==0).sum())} negative")
    print(f"Balanced: {len(pos)} positive / {len(matched_neg)} negative")
    print(f"Median matched descriptor distance (z-space): {match_diag['descriptor_distance_z'].median():.3f}")
    print(f"Median matched ECFP4 Tanimoto: {match_diag['matched_ECFP4_tanimoto'].median():.3f}")
    print(f"Median nearest-negative ECFP4 Tanimoto: {nearest['max_ECFP4_tanimoto'].median():.3f}")
    print(f"Max nearest-negative ECFP4 Tanimoto: {nearest['max_ECFP4_tanimoto'].max():.3f}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Audit frozen scaffold CV splits and baseline OOF predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.metrics import roc_auc_score, average_precision_score

DATASETS = {
    "strict": (
        "data/kmol/umami_strict.csv",
        "data/kmol/scaffold_cv/scaffold_5fold_splits.json",
    ),
    "full": (
        "data/kmol/full_primary/umami_full_primary.csv",
        "data/kmol/full_primary/scaffold_cv/scaffold_5fold_splits.json",
    ),
    "matched": (
        "data/kmol/physchem_matched/umami_physchem_matched.csv",
        "data/kmol/physchem_matched/scaffold_cv/scaffold_5fold_splits.json",
    ),
}

OUT = Path("results/jcim_scaffold_audit")
OUT.mkdir(parents=True, exist_ok=True)


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, p), average_precision_score(y, p)


def load_dataset(name):
    data_path, split_path = DATASETS[name]
    df = pd.read_csv(data_path).reset_index(drop=True)
    splits = json.loads(Path(split_path).read_text())

    folds = np.full(len(df), -1, dtype=int)
    for key, indices in splits.items():
        f = int(key.split("_")[-1])
        ids = np.asarray(indices, dtype=int)
        if np.any(ids < 0) or np.any(ids >= len(df)):
            raise ValueError(f"{name}: invalid indices")
        if np.any(folds[ids] != -1):
            raise ValueError(f"{name}: duplicate fold assignment")
        folds[ids] = f

    if np.any(folds == -1):
        raise ValueError(f"{name}: unassigned rows")

    scaffolds = []
    for s in df["smiles"]:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {s}")
        scaffolds.append(
            MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        )

    df["row_id"] = np.arange(len(df))
    df["fold"] = folds
    df["scaffold"] = scaffolds
    df["acyclic"] = df["scaffold"].eq("")
    df["label"] = df["umami"].astype(int)

    leakage = df.groupby("scaffold")["fold"].nunique()
    if (leakage > 1).any():
        raise ValueError(f"{name}: scaffold leakage detected")

    return df


def audit_folds(name, df):
    rows = []

    for fold, g in df.groupby("fold"):
        rows.append({
            "dataset": name,
            "fold": int(fold),
            "n": len(g),
            "positive": int(g["label"].sum()),
            "negative": int((g["label"] == 0).sum()),
            "acyclic_total": int(g["acyclic"].sum()),
            "acyclic_positive": int(
                (g["acyclic"] & (g["label"] == 1)).sum()
            ),
            "acyclic_negative": int(
                (g["acyclic"] & (g["label"] == 0)).sum()
            ),
            "cyclic_positive": int(
                ((~g["acyclic"]) & (g["label"] == 1)).sum()
            ),
            "cyclic_negative": int(
                ((~g["acyclic"]) & (g["label"] == 0)).sum()
            ),
            "unique_scaffolds": int(g["scaffold"].nunique()),
        })

    return pd.DataFrame(rows)


def audit_predictions(name, df):
    path = Path(
        f"results/jcim_baselines/{name}/oof_predictions.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    oof = pd.read_csv(path)
    results = []
    prediction_rows = []

    for model, g in oof.groupby("model"):
        g = g.sort_values("row_id").reset_index(drop=True)

        if len(g) != len(df):
            raise ValueError(f"{name}/{model}: incorrect row count")
        if not np.array_equal(
            g["row_id"].to_numpy(), df["row_id"].to_numpy()
        ):
            raise ValueError(f"{name}/{model}: row IDs differ")
        if not np.array_equal(
            g["label"].astype(int).to_numpy(),
            df["label"].to_numpy()
        ):
            raise ValueError(f"{name}/{model}: labels differ")
        if not np.array_equal(
            g["fold"].astype(int).to_numpy(),
            df["fold"].to_numpy()
        ):
            raise ValueError(f"{name}/{model}: folds differ")
        if not np.array_equal(
            g["smiles"].astype(str).to_numpy(),
            df["smiles"].astype(str).to_numpy()
        ):
            raise ValueError(f"{name}/{model}: SMILES differ")

        g["acyclic"] = df["acyclic"].to_numpy()
        prediction_rows.append(g)

        subsets = {
            "all": np.ones(len(g), dtype=bool),
            "cyclic_only": ~g["acyclic"].to_numpy(),
            "acyclic_only": g["acyclic"].to_numpy(),
        }

        for fold in sorted(df["fold"].unique()):
            subsets[f"fold_{fold}"] = (
                g["fold"].to_numpy() == fold
            )
            subsets[f"excluding_fold_{fold}"] = (
                g["fold"].to_numpy() != fold
            )

        for subset_name, mask in subsets.items():
            sub = g.loc[mask]
            auc, ap = metrics(
                sub["label"], sub["probability"]
            )
            results.append({
                "dataset": name,
                "model": model,
                "subset": subset_name,
                "n": len(sub),
                "positive": int(sub["label"].sum()),
                "negative": int((sub["label"] == 0).sum()),
                "roc_auc": auc,
                "pr_auc": ap,
            })

    return (
        pd.DataFrame(results),
        pd.concat(prediction_rows, ignore_index=True),
    )


def main():
    all_folds = []
    all_metrics = []
    all_predictions = []

    for name in DATASETS:
        print(f"\n{'=' * 65}")
        print(name.upper())

        df = load_dataset(name)
        fold_summary = audit_folds(name, df)
        metric_summary, predictions = audit_predictions(name, df)

        df.to_csv(
            OUT / f"{name}_molecule_audit.csv", index=False
        )

        all_folds.append(fold_summary)
        all_metrics.append(metric_summary)
        all_predictions.append(predictions)

        print("\nFold composition:")
        print(fold_summary.to_string(index=False))

        print("\nOOF subset performance:")
        selected = metric_summary[
            metric_summary["subset"].isin([
                "all", "cyclic_only", "acyclic_only"
            ])
        ]
        print(selected.to_string(index=False))

    folds = pd.concat(all_folds, ignore_index=True)
    scores = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)

    folds.to_csv(OUT / "fold_composition.csv", index=False)
    scores.to_csv(OUT / "subset_performance.csv", index=False)
    predictions.to_csv(
        OUT / "annotated_oof_predictions.csv", index=False
    )

    # Matched-pair audit: inspect source columns before deciding
    # whether pairs can be mapped to exact molecule IDs.
    pair_path = Path("data/analysis/physchem_matched_pairs.csv")
    if pair_path.exists():
        pairs = pd.read_csv(pair_path)
        print("\n=== MATCHED PAIR SOURCE ===")
        print("Rows:", len(pairs))
        print("Columns:", pairs.columns.tolist())
        print(pairs.head(3).to_string(index=False))

    print("\nSaved audit results to:", OUT)


if __name__ == "__main__":
    main()

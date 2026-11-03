
#!/usr/bin/env python3
"""Sensitivity analysis for the dominant umami scaffold.

Uses frozen out-of-fold predictions.
No model refitting or hyperparameter tuning.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("results/jcim_scaffold_audit")
OUT = ROOT / "dominant_scaffold"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "O=c1[nH]cnc2c1ncn2C1CCCO1"

DATASETS = ["strict", "full", "matched"]


def evaluate(df):
    y = df["label"].astype(int).to_numpy()
    p = df["probability"].astype(float).to_numpy()

    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    return roc_auc_score(y, p), average_precision_score(y, p)


results = []
fold_info = []

for dataset in DATASETS:
    mol = pd.read_csv(
        ROOT / f"{dataset}_molecule_audit.csv",
        keep_default_na=False
    )
    pred = pd.read_csv(
        f"results/jcim_baselines/{dataset}/oof_predictions.csv"
    )

    mol["row_id"] = mol["row_id"].astype(int)
    mol["fold"] = mol["fold"].astype(int)
    mol["label"] = mol["label"].astype(int)

    target = mol["scaffold"] == TARGET

    target_folds = mol.loc[target, "fold"].unique()
    if len(target_folds) != 1:
        raise ValueError(
            f"{dataset}: target scaffold spans multiple folds"
        )

    target_fold = int(target_folds[0])

    print(f"\n=== {dataset.upper()} ===")
    print("Target fold:", target_fold)
    print("Target scaffold compounds:", int(target.sum()))
    print(
        "Positive:",
        int((target & (mol["label"] == 1)).sum())
    )
    print(
        "Negative:",
        int((target & (mol["label"] == 0)).sum())
    )

    fold_info.append({
        "dataset": dataset,
        "target_fold": target_fold,
        "target_total": int(target.sum()),
        "target_positive": int(
            (target & (mol["label"] == 1)).sum()
        ),
        "target_negative": int(
            (target & (mol["label"] == 0)).sum()
        ),
    })

    for model, g in pred.groupby("model"):
        g = g.merge(
            mol[["row_id", "scaffold", "fold", "label"]],
            on="row_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_audit")
        )

        if len(g) != len(mol):
            raise ValueError("Unexpected prediction count")

        if not np.array_equal(
            g["label"].to_numpy().astype(int),
            g["label_audit"].to_numpy().astype(int)
        ):
            raise ValueError("Label mismatch")

        if not np.array_equal(
            g["fold"].to_numpy().astype(int),
            g["fold_audit"].to_numpy().astype(int)
        ):
            raise ValueError("Fold mismatch")

        subsets = {
            "all": np.ones(len(g), dtype=bool),
            "exclude_32_positives": ~(
                (g["scaffold"] == TARGET) &
                (g["label"] == 1)
            ),
            "exclude_target_scaffold": (
                g["scaffold"] != TARGET
            ),
            "exclude_target_fold": (
                g["fold"] != target_fold
            ),
        }

        for subset_name, mask in subsets.items():
            sub = g.loc[mask]
            auc, ap = evaluate(sub)

            results.append({
                "dataset": dataset,
                "model": model,
                "subset": subset_name,
                "n": len(sub),
                "positive": int(sub["label"].sum()),
                "negative": int((sub["label"] == 0).sum()),
                "roc_auc": auc,
                "pr_auc": ap,
            })

result_df = pd.DataFrame(results)
fold_df = pd.DataFrame(fold_info)

result_df.to_csv(
    OUT / "sensitivity_performance.csv",
    index=False
)
fold_df.to_csv(
    OUT / "target_scaffold_composition.csv",
    index=False
)

print("\n=== SENSITIVITY PERFORMANCE ===")
print(result_df.to_string(index=False))

print("\nSaved to:", OUT)

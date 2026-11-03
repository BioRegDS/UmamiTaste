
#!/usr/bin/env python3
"""Molecular ML baselines using frozen kMoL scaffold folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

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


def load_data(name):
    csv_path, split_path = DATASETS[name]
    df = pd.read_csv(csv_path)
    splits = json.loads(Path(split_path).read_text())

    assert {"smiles", "umami"}.issubset(df.columns)
    y = df["umami"].astype(int).to_numpy()
    assert set(np.unique(y)) == {0, 1}

    fold_ids = np.full(len(df), -1, dtype=int)

    for key, indices in splits.items():
        fold = int(key.split("_")[-1])
        idx = np.asarray(indices, dtype=int)
        assert np.all((idx >= 0) & (idx < len(df)))
        assert np.all(fold_ids[idx] == -1), "Duplicate fold assignment"
        fold_ids[idx] = fold

    assert np.all(fold_ids >= 0)
    assert set(np.unique(fold_ids)) == set(range(5))

    return df, y, fold_ids


def make_features(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    )
    descriptor_list = Descriptors._descList
    fp_rows, desc_rows, scaffolds = [], [], []

    for s in smiles:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {s}")

        arr = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(
            generator.GetFingerprint(mol), arr
        )
        fp_rows.append(arr)

        values = []
        for _, func in descriptor_list:
            try:
                values.append(float(func(mol)))
            except Exception:
                values.append(np.nan)
        desc_rows.append(values)

        scaffolds.append(
            MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        )

    fp = np.asarray(fp_rows, dtype=np.float32)
    desc = np.asarray(desc_rows, dtype=np.float64)
    desc[~np.isfinite(desc)] = np.nan

    return fp, desc, scaffolds


def build_model(name, y_train):
    if name == "RF_Morgan":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=n_neg / n_pos,
        random_state=42,
        n_jobs=4,
        tree_method="hist",
    )

    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        model,
    )


def score(y, prob):
    if len(np.unique(y)) != 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan}
    return {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
    }


def run_dataset(name, output_root):
    df, y, folds = load_data(name)
    fp, desc, scaffolds = make_features(df["smiles"])

    # Every scaffold, including the empty acyclic scaffold,
    # must occur in only one test fold.
    audit = pd.DataFrame({
        "scaffold": scaffolds, "fold": folds
    })
    assert audit.groupby("scaffold")["fold"].nunique().max() == 1, (
        "Scaffold leakage detected"
    )

    features = {
        "RF_Morgan": fp,
        "XGB_RDKit": desc,
        "XGB_Morgan_RDKit": np.hstack([fp, desc]),
    }

    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    oof_rows = []

    print(f"\n=== {name.upper()} ===")
    print(f"N={len(df)}, positives={int(y.sum())}")

    for fold in sorted(np.unique(folds)):
        train_idx = np.flatnonzero(folds != fold)
        test_idx = np.flatnonzero(folds == fold)

        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) != 2:
            raise ValueError(f"{name} fold {fold}: one-class training set")

        print(
            f"Fold {fold}: train={len(train_idx)}, "
            f"test={len(test_idx)}, test_pos={int(y_test.sum())}, "
            f"acyclic={sum(scaffolds[i] == '' for i in test_idx)}"
        )

        for model_name, X in features.items():
            model = build_model(model_name, y_train)
            model.fit(X[train_idx], y_train)
            prob = model.predict_proba(X[test_idx])[:, 1]

            metrics = score(y_test, prob)
            fold_rows.append({
                "dataset": name,
                "model": model_name,
                "fold": int(fold),
                "n_test": len(test_idx),
                "n_positive": int(y_test.sum()),
                **metrics,
            })

            for i, p in zip(test_idx, prob):
                oof_rows.append({
                    "dataset": name,
                    "model": model_name,
                    "row_id": int(i),
                    "smiles": df.iloc[i]["smiles"],
                    "label": int(y[i]),
                    "fold": int(fold),
                    "probability": float(p),
                })

            print(
                f"  {model_name}: "
                f"AUC={metrics['roc_auc']:.4f}, "
                f"AP={metrics['pr_auc']:.4f}"
            )

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)

    summary = []
    for model_name, group in oof_df.groupby("model"):
        group = group.sort_values("row_id")
        assert group["row_id"].tolist() == list(range(len(df)))
        metrics = score(
            group["label"].to_numpy(),
            group["probability"].to_numpy(),
        )
        summary.append({
            "dataset": name,
            "model": model_name,
            "n": len(group),
            "oof_roc_auc": metrics["roc_auc"],
            "oof_pr_auc": metrics["pr_auc"],
        })

    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(
        output_dir / "summary.csv", index=False
    )

    print("\nOOF summary:")
    print(pd.DataFrame(summary).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["strict", "full", "matched", "all"],
        default="all",
    )
    parser.add_argument(
        "--output-dir", default="results/jcim_baselines"
    )
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        run_dataset(name, Path(args.output_dir))


if __name__ == "__main__":
    main()

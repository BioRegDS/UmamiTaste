#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--weight-json", required=True)
    ap.add_argument("--output-config", required=True)
    ap.add_argument("--output-path", default="data/logs/umami_full_primary_scaffold_cv/")
    ap.add_argument("--cache-location", default="data/cache/umami_full_primary_scaffold_cv/")
    args = ap.parse_args()

    with open(args.base_config) as fh:
        cfg = json.load(fh)
    with open(args.weight_json) as fh:
        w = json.load(fh)

    cfg = copy.deepcopy(cfg)
    cfg["loader"]["input_path"] = args.data_csv
    cfg["splitter"] = {
        "type": "precomputed",
        "splits": {f"fold_{i}": 0.2 for i in range(5)},
        "split_path": args.split_json,
    }
    cfg["cross_validation_folds"] = 5
    cfg["criterion"] = {
        "type": "umami_criterions.PosWeightedBCEWithLogitsLoss",
        "pos_weight": float(w["pos_weight"]),
    }
    cfg["output_path"] = args.output_path
    cfg["cache_location"] = args.cache_location
    cfg["target_metric"] = "pr_auc"
    cfg["test_metrics"] = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "cohen_kappa",
    ]

    Path(args.output_config).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_config, "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"Wrote: {args.output_config}")
    print(f"pos_weight: {w['pos_weight']:.6f}")
    print("target_metric: pr_auc")


if __name__ == "__main__":
    main()

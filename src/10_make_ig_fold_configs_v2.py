#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--checkpoint-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    with open(args.base_config) as fh:
        base = json.load(fh)

    with open(args.checkpoint_manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Created IG configs:")
    for row in rows:
        fold = int(row["fold"])
        checkpoint = Path(row["selected_checkpoint"])

        cfg = json.loads(json.dumps(base))
        fold_name = f"fold_{fold}"
        fold_dir = outdir / fold_name
        image_dir = fold_dir / "images"
        fold_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        cfg["checkpoint_path"] = str(checkpoint)
        cfg["test_split"] = fold_name
        cfg["output_path"] = str(fold_dir / "kmol_output")
        cfg["visualizer"] = {
            "type": "iig",
            "is_binary_classification": True,
            "is_multitask": False,
            "targets": [0],
            "mapping_file_path": str(fold_dir / "mapping.csv"),
            "graph_input_key": "graph",
            "sketcher": {
                "type": "kmol.visualization.sketchers.RdkitSketcher",
                "output_path": str(image_dir)
            }
        }

        out = outdir / f"ig_{fold_name}.json"
        with out.open("w") as fh:
            json.dump(cfg, fh, indent=2)

        print(
            f"  fold {fold}: epoch {row['selected_epoch']} -> {out}\n"
            f"           checkpoint: {checkpoint} "
            f"[{'OK' if checkpoint.exists() else 'MISSING'}]"
        )

if __name__ == "__main__":
    main()

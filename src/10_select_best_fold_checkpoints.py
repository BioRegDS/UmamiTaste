#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

PATTERN = re.compile(
    r"Restoring from Checkpoint:\s+(?P<path>.+?/\.(?P<fold>\d+)/.+?/checkpoint\.(?P<epoch>\d+)\.pt)"
)

def main():
    p = argparse.ArgumentParser(
        description="Parse a kMoL standard_cv log and recover the checkpoint finally selected for each fold."
    )
    p.add_argument("--log", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    log_path = Path(args.log)
    restores = {i: [] for i in range(args.folds)}

    with log_path.open(errors="ignore") as fh:
        for line_no, line in enumerate(fh, start=1):
            m = PATTERN.search(line)
            if not m:
                continue
            fold = int(m.group("fold"))
            restores.setdefault(fold, []).append({
                "line_no": line_no,
                "checkpoint_path": m.group("path"),
                "epoch": int(m.group("epoch")),
            })

    rows = []
    for fold in range(args.folds):
        items = restores.get(fold, [])
        if not items:
            raise RuntimeError(f"No restored checkpoints found for fold {fold}")

        selected = items[-1]
        candidate_epochs = [x["epoch"] for x in items[:-1]]
        rows.append({
            "fold": fold,
            "selected_epoch": selected["epoch"],
            "selected_checkpoint": selected["checkpoint_path"],
            "selected_restore_line": selected["line_no"],
            "n_restore_events": len(items),
            "max_candidate_epoch_seen": max(candidate_epochs) if candidate_epochs else "",
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Selected checkpoints:")
    for r in rows:
        exists = Path(r["selected_checkpoint"]).exists()
        print(
            f'  fold {r["fold"]}: epoch {r["selected_epoch"]} -> '
            f'{r["selected_checkpoint"]} [{"OK" if exists else "MISSING"}]'
        )
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()

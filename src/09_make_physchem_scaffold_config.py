#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-config', required=True)
    ap.add_argument('--data-csv', required=True)
    ap.add_argument('--split-json', required=True)
    ap.add_argument('--output-config', required=True)
    ap.add_argument('--output-path', default='data/logs/umami_physchem_matched_scaffold_cv/')
    ap.add_argument('--cache-location', default='data/cache/umami_physchem_matched_scaffold_cv/')
    args = ap.parse_args()

    with open(args.base_config) as f:
        cfg = json.load(f)

    cfg['loader']['input_path'] = args.data_csv
    cfg['splitter'] = {
        'type': 'precomputed',
        'splits': {f'fold_{i}': 0.2 for i in range(5)},
        'split_path': args.split_json,
    }

    # Balanced 1:1 data: use the same criterion structure as the working
    # strict baseline config. kMoL's MaskedLoss wraps torch BCEWithLogitsLoss.
    cfg['criterion'] = {
        'type': 'kmol.model.criterions.MaskedLoss',
        'loss': {
            'type': 'torch.nn.BCEWithLogitsLoss'
        }
    }

    cfg['output_path'] = args.output_path
    cfg['cache_location'] = args.cache_location
    cfg['target_metric'] = 'roc_auc'

    out = Path(args.output_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f'Saved: {out}')
    print('Criterion:', cfg['criterion'])


if __name__ == '__main__':
    main()

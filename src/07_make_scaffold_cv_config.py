#!/usr/bin/env python3
"""Create a kMoL scaffold-CV config from an existing working baseline config."""
import argparse, json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--base-config', required=True)
p.add_argument('--split-json', required=True)
p.add_argument('--output-config', required=True)
p.add_argument('--output-path', default='data/logs/umami_strict_scaffold_cv/')
a=p.parse_args()

with open(a.base_config) as f:
    cfg=json.load(f)

cfg['splitter']={
    'type':'precomputed',
    'splits': {f'fold_{i}': 0.2 for i in range(5)},
    'split_path': a.split_json
}
cfg['cross_validation_folds']=5
cfg['output_path']=a.output_path
cfg['checkpoint_path']=None
# Separate cache namespace; preprocessing itself is identical, but keeping experiment artifacts isolated is clearer.
cfg['cache_location']='data/cache/umami_strict_scaffold_cv/'

Path(a.output_config).parent.mkdir(parents=True, exist_ok=True)
with open(a.output_config,'w') as f:
    json.dump(cfg,f,indent=2)
print(f'Wrote {a.output_config}')

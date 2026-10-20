# UmamiTaste

Code and configuration files accompanying a study of graph neural network (GNN)-based umami taste prediction and interpretation.

## Repository scope

This repository is intended to document data processing, model configuration, scaffold-disjoint cross-validation, Integrated Gradients (IG) analysis, functional-group comparisons, matched-pair analysis, and figure generation. It does **not** include pretrained model checkpoints, training logs, bulk IG outputs, or the source ChemTastesDB Excel workbook.

## Data

The analysis uses ChemTastesDB-derived small-molecule data. Obtain the source dataset from its original provider, subject to its terms of use. The preprocessing scripts expect `data/raw/ChemTastesDB_database.xlsx` as input; this file is **not included** here.

Three study datasets are considered: Strict (283 molecules), Full primary (2,394 molecules), and physicochemical-property-matched (148 molecules). Exact five-fold scaffold assignments are provided under `data/kmol/`.

## Software

The original analysis used Python 3.11.16, RDKit 2022.09.4, PyTorch 1.13.1, PyTorch Geometric 2.3.0, and kMoL at commit `92715aaa00a909f62b9c36f5fdd9bd1fb505dba3`. Reproducing the original environment may require additional dependencies. An independently tested environment installation procedure has not yet been provided.

## Workflow overview

1. Dataset preparation: `src/01_prepare_chemtastesdb.py` through `src/04_prepare_training_sets.py`.
2. kMoL input preparation and scaffold CV configuration: scripts `06_` through `09_` in `src/`.
3. Train/evaluate the GNN using the relevant files in `configs/kmol/` and the original kMoL installation.
4. Export out-of-fold atom-level IG using `src/10_export_signed_ig_v2.py`, then merge and analyze with `src/11_merge_and_annotate_ig.py`, `src/12_compare_functional_group_ig.py`, and `src/13_analyze_matched_pairs_ig_v2.py`.
5. Generate study figures with the `14_`, `15_`, `16_`, and `18_` scripts, and matched-pair sensitivity results with `17_matched_pair_sensitivity.py`.

**Important:** This is a workflow outline, not a fully tested end-to-end execution recipe. Some scripts require intermediate outputs and trained checkpoints that are intentionally not distributed. The performance figure script reads original training logs; only the corresponding small summary CSV files are distributed.

## Scaffold split files

- Strict: `data/kmol/scaffold_cv/`
- Full primary: `data/kmol/full_primary/scaffold_cv/`
- Physicochemical matched: `data/kmol/physchem_matched/scaffold_cv/`

Each directory contains `scaffold_5fold_splits.json` and `scaffold_fold_assignments.csv`.

## Notes on reproducibility

Scripts retain their original relative paths and filenames. Model weights and intermediate analysis outputs are not provided. Some generated IG configuration files may contain local absolute paths and should be checked before use on another machine.

## Citation

Please cite the associated article when available. [Add final article citation/DOI after publication.]

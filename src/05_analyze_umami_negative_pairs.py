#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem import rdFingerprintGenerator

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

def get_morgan_fp(mol):
    return morgan_generator.GetFingerprint(mol)

def parse_args():
    p = argparse.ArgumentParser(description='Analyze umami-positive vs negative molecular pairs using ECFP Tanimoto and MCS.')
    p.add_argument('--positive', required=True, help='CSV containing positive molecules')
    p.add_argument('--negative', required=True, help='CSV containing negative molecules')
    p.add_argument('--output-dir', required=True, help='Output directory')
    p.add_argument('--smiles-col', default='ml_SMILES', help='SMILES column name')
    p.add_argument('--name-col', default='Name', help='Name column name')
    p.add_argument('--id-col', default='ID', help='ID column name')
    p.add_argument('--radius', type=int, default=2, help='Morgan fingerprint radius')
    p.add_argument('--nbits', type=int, default=2048, help='Morgan fingerprint bit length')
    p.add_argument('--mcs-threshold', type=float, default=0.50, help='Run MCS for pairs with Tanimoto >= threshold')
    p.add_argument('--mcs-timeout', type=int, default=10, help='MCS timeout per pair in seconds')
    p.add_argument('--top-k-per-positive', type=int, default=10, help='Store top-K negative neighbors for each positive')
    return p.parse_args()


def mol_from_smiles(s):
    if pd.isna(s):
        return None
    try:
        return Chem.MolFromSmiles(str(s))
    except Exception:
        return None


def fp_from_mol(mol, radius, nbits):
    if mol is None:
        return None
    return morgan_generator.GetFingerprint(mol)


def atom_symbols(mol, atom_indices):
    return ';'.join(f'{i}:{mol.GetAtomWithIdx(i).GetSymbol()}' for i in atom_indices)


def bond_ids_within_atoms(mol, atom_indices):
    aset = set(atom_indices)
    out = []
    for b in mol.GetBonds():
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        if a1 in aset and a2 in aset:
            out.append(b.GetIdx())
    return out


def analyze_mcs(pos_mol, neg_mol, timeout):
    result = {
        'mcs_num_atoms': 0,
        'mcs_num_bonds': 0,
        'mcs_smarts': '',
        'mcs_canceled': False,
        'pos_mcs_atom_indices': '',
        'neg_mcs_atom_indices': '',
        'pos_specific_atom_indices': '',
        'neg_specific_atom_indices': '',
        'pos_specific_atoms': '',
        'neg_specific_atoms': '',
        'pos_specific_bond_indices': '',
        'neg_specific_bond_indices': '',
        'pos_fraction_in_mcs': math.nan,
        'neg_fraction_in_mcs': math.nan,
    }
    try:
        mcs = rdFMCS.FindMCS(
            [pos_mol, neg_mol],
            timeout=timeout,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
        result['mcs_canceled'] = bool(mcs.canceled)
        result['mcs_num_atoms'] = int(mcs.numAtoms)
        result['mcs_num_bonds'] = int(mcs.numBonds)
        result['mcs_smarts'] = mcs.smartsString or ''
        if mcs.numAtoms == 0 or not mcs.smartsString:
            return result
        patt = Chem.MolFromSmarts(mcs.smartsString)
        if patt is None:
            return result
        pos_match = pos_mol.GetSubstructMatch(patt)
        neg_match = neg_mol.GetSubstructMatch(patt)
        pos_match = tuple(pos_match) if pos_match else tuple()
        neg_match = tuple(neg_match) if neg_match else tuple()
        pos_specific = sorted(set(range(pos_mol.GetNumAtoms())) - set(pos_match))
        neg_specific = sorted(set(range(neg_mol.GetNumAtoms())) - set(neg_match))
        result['pos_mcs_atom_indices'] = ';'.join(map(str, pos_match))
        result['neg_mcs_atom_indices'] = ';'.join(map(str, neg_match))
        result['pos_specific_atom_indices'] = ';'.join(map(str, pos_specific))
        result['neg_specific_atom_indices'] = ';'.join(map(str, neg_specific))
        result['pos_specific_atoms'] = atom_symbols(pos_mol, pos_specific)
        result['neg_specific_atoms'] = atom_symbols(neg_mol, neg_specific)
        result['pos_specific_bond_indices'] = ';'.join(map(str, bond_ids_within_atoms(pos_mol, pos_specific)))
        result['neg_specific_bond_indices'] = ';'.join(map(str, bond_ids_within_atoms(neg_mol, neg_specific)))
        result['pos_fraction_in_mcs'] = mcs.numAtoms / max(1, pos_mol.GetNumAtoms())
        result['neg_fraction_in_mcs'] = mcs.numAtoms / max(1, neg_mol.GetNumAtoms())
    except Exception:
        pass
    return result


def safe_get(row, col, default=''):
    return row[col] if col in row and not pd.isna(row[col]) else default


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    analysis_dir = outdir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pos = pd.read_csv(args.positive)
    neg = pd.read_csv(args.negative)

    required = [args.smiles_col]
    for c in required:
        if c not in pos.columns or c not in neg.columns:
            raise ValueError(f'Missing required column: {c}')

    pos_records = []
    for idx, row in pos.iterrows():
        mol = mol_from_smiles(row[args.smiles_col])
        if mol is None:
            continue
        fp = fp_from_mol(mol, args.radius, args.nbits)
        pos_records.append((idx, row, mol, fp))

    neg_records = []
    for idx, row in neg.iterrows():
        mol = mol_from_smiles(row[args.smiles_col])
        if mol is None:
            continue
        fp = fp_from_mol(mol, args.radius, args.nbits)
        neg_records.append((idx, row, mol, fp))

    all_pairs = []
    topk_rows = []
    threshold_counts = {0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0}
    mcs_candidates = []

    for p_idx, p_row, p_mol, p_fp in pos_records:
        sims = []
        for n_idx, n_row, n_mol, n_fp in neg_records:
            sim = float(DataStructs.TanimotoSimilarity(p_fp, n_fp))
            for th in threshold_counts:
                if sim >= th:
                    threshold_counts[th] += 1
            record = {
                'positive_index': p_idx,
                'positive_id': safe_get(p_row, args.id_col, p_idx),
                'positive_name': safe_get(p_row, args.name_col, f'positive_{p_idx}'),
                'positive_smiles': p_row[args.smiles_col],
                'negative_index': n_idx,
                'negative_id': safe_get(n_row, args.id_col, n_idx),
                'negative_name': safe_get(n_row, args.name_col, f'negative_{n_idx}'),
                'negative_smiles': n_row[args.smiles_col],
                'tanimoto_ecfp4': sim,
            }
            all_pairs.append(record)
            sims.append(record)
            if sim >= args.mcs_threshold:
                mcs_candidates.append((record, p_mol, n_mol))

        sims.sort(key=lambda x: x['tanimoto_ecfp4'], reverse=True)
        for rank, rec in enumerate(sims[:args.top_k_per_positive], start=1):
            rr = dict(rec)
            rr['rank_within_positive'] = rank
            topk_rows.append(rr)

    all_pairs_df = pd.DataFrame(all_pairs)
    all_pairs_df.to_csv(analysis_dir / 'all_positive_negative_tanimoto_pairs.csv', index=False)

    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(analysis_dir / 'top_similar_negatives_per_positive.csv', index=False)

    summary_rows = []
    total_pairs = len(all_pairs_df)
    for th in sorted(threshold_counts):
        count = threshold_counts[th]
        summary_rows.append({
            'threshold': th,
            'pair_count': count,
            'pair_fraction': count / total_pairs if total_pairs else math.nan,
        })
    pd.DataFrame(summary_rows).to_csv(analysis_dir / 'tanimoto_threshold_summary.csv', index=False)

    mcs_rows = []
    for rec, p_mol, n_mol in mcs_candidates:
        mcs_info = analyze_mcs(p_mol, n_mol, args.mcs_timeout)
        rr = dict(rec)
        rr.update(mcs_info)
        mcs_rows.append(rr)
    mcs_df = pd.DataFrame(mcs_rows)
    if not mcs_df.empty:
        mcs_df = mcs_df.sort_values(['tanimoto_ecfp4', 'mcs_num_atoms'], ascending=[False, False])
    mcs_df.to_csv(analysis_dir / 'high_similarity_pairs_with_mcs.csv', index=False)

    if not all_pairs_df.empty:
        nearest = all_pairs_df.sort_values(['positive_index', 'tanimoto_ecfp4'], ascending=[True, False]) \
                              .groupby('positive_index', as_index=False).head(1)
        nearest.to_csv(analysis_dir / 'nearest_negative_per_positive.csv', index=False)

    print(f'Positive molecules parsed: {len(pos_records)}')
    print(f'Negative molecules parsed: {len(neg_records)}')
    print(f'All pairs: {total_pairs}')
    for th in sorted(threshold_counts):
        print(f'Tanimoto >= {th:.1f}: {threshold_counts[th]}')
    print(f'MCS analyzed pairs (threshold >= {args.mcs_threshold:.2f}): {len(mcs_rows)}')
    if not all_pairs_df.empty:
        print(f'Median Tanimoto across all pairs: {all_pairs_df.tanimoto_ecfp4.median():.3f}')
        print(f'Maximum Tanimoto: {all_pairs_df.tanimoto_ecfp4.max():.3f}')
    print(f'Outputs written to: {analysis_dir}')


if __name__ == '__main__':
    main()

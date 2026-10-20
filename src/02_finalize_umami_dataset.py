#!/usr/bin/env python3
"""
Finalize the low-molecular-weight umami-positive dataset.

Input
-----
umami_small_molecules.csv produced by 01_prepare_chemtastesdb.py

Main operations
---------------
1. Parse each SMILES into disconnected fragments.
2. Ignore simple counterions/metals when identifying the parent structure.
3. Distinguish true mixtures from salts/stoichiometric repeats:
   - multiple identical substantive parent fragments -> keep one parent
   - multiple different substantive parent fragments -> exclude as mixture
4. Standardize the retained parent with RDKit MolStandardize:
   cleanup -> fragment parent -> uncharge -> canonical isomeric SMILES.
5. Recalculate parent molecular descriptors.
6. Merge records that become identical after standardization.
7. Save a complete audit trail.

Important
---------
The original chemical identity and original SMILES are never discarded.
The standardized parent SMILES is intended for molecular-structure ML/GNN input.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


# Fragments with <= 1 heavy atom are treated as simple counterions by default.
# Examples: [Na+], [K+], [Ca+2], [Mg+2], [Cl-].
MIN_SUBSTANTIVE_HEAVY_ATOMS = 2


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def safe_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return ""
        return canonical_smiles(scaf)
    except Exception:
        return ""


def fragment_signature(mol: Chem.Mol) -> str:
    """Neutralized canonical signature used only to compare substantive fragments."""
    try:
        m = rdMolStandardize.Cleanup(mol)
        m = rdMolStandardize.Uncharger().uncharge(m)
        return canonical_smiles(m)
    except Exception:
        return canonical_smiles(mol)


def split_fragments(mol: Chem.Mol) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    substantive, counterions = [], []
    for frag in frags:
        heavy = frag.GetNumHeavyAtoms()
        if heavy >= MIN_SUBSTANTIVE_HEAVY_ATOMS:
            substantive.append(frag)
        else:
            counterions.append(frag)
    return substantive, counterions


def standardize_parent(parent: Chem.Mol) -> Chem.Mol:
    """Standardize a single parent fragment for ML use."""
    m = rdMolStandardize.Cleanup(parent)
    # Defensive: retain the largest meaningful fragment if anything remains disconnected.
    m = rdMolStandardize.FragmentParent(m)
    # Convert salts such as glutamate anion to a neutral parent representation where possible.
    m = rdMolStandardize.Uncharger().uncharge(m)
    Chem.SanitizeMol(m)
    return m


def describe_mol(mol: Chem.Mol) -> Dict[str, object]:
    return {
        "parent_MW": round(Descriptors.MolWt(mol), 3),
        "parent_HeavyAtomCount": int(mol.GetNumHeavyAtoms()),
        "parent_HBD": int(Lipinski.NumHDonors(mol)),
        "parent_HBA": int(Lipinski.NumHAcceptors(mol)),
        "parent_RotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
        "parent_TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "parent_FormalCharge": int(Chem.GetFormalCharge(mol)),
        "parent_MurckoScaffold": safe_scaffold_smiles(mol),
    }


def process_row(row: pd.Series) -> Dict[str, object]:
    out = row.to_dict()
    smi = str(row.get("canonical_SMILES", "") or "").strip()

    out.update({
        "final_action": "",
        "exclusion_reason": "",
        "n_fragments": 0,
        "n_substantive_fragments": 0,
        "n_unique_substantive_parents": 0,
        "counterion_SMILES": "",
        "substantive_fragment_SMILES": "",
        "parent_SMILES": "",
        "standardization_note": "",
    })

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        out["final_action"] = "EXCLUDE"
        out["exclusion_reason"] = "SMILES parse failure"
        return out

    all_frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    substantive, counterions = split_fragments(mol)
    out["n_fragments"] = len(all_frags)
    out["n_substantive_fragments"] = len(substantive)
    out["counterion_SMILES"] = " | ".join(canonical_smiles(x) for x in counterions)
    out["substantive_fragment_SMILES"] = " | ".join(canonical_smiles(x) for x in substantive)

    if len(substantive) == 0:
        out["final_action"] = "EXCLUDE"
        out["exclusion_reason"] = "No substantive molecular fragment"
        return out

    signatures = [fragment_signature(x) for x in substantive]
    unique_signatures = sorted(set(signatures))
    out["n_unique_substantive_parents"] = len(unique_signatures)

    # Different substantive molecular species = mixture, not a single-molecule observation.
    if len(unique_signatures) > 1:
        out["final_action"] = "EXCLUDE"
        out["exclusion_reason"] = "Multiple different substantive molecular species (mixture)"
        return out

    # If the same substantive species occurs multiple times (e.g. calcium diglutamate),
    # retain one representative parent structure.
    representative = substantive[0]
    try:
        parent = standardize_parent(representative)
    except Exception as exc:
        out["final_action"] = "EXCLUDE"
        out["exclusion_reason"] = f"Standardization failure: {type(exc).__name__}"
        return out

    parent_smi = canonical_smiles(parent)
    out["parent_SMILES"] = parent_smi

    notes = []
    if len(all_frags) > 1:
        notes.append("desalted")
    if len(substantive) > 1 and len(unique_signatures) == 1:
        notes.append("collapsed repeated identical parent fragments")
    if parent_smi != smi:
        notes.append("parent representation standardized/uncharged")

    out["standardization_note"] = "; ".join(notes) if notes else "unchanged"
    out["final_action"] = "STANDARDIZE" if notes else "KEEP"
    out.update(describe_mol(parent))
    return out


def aggregate_duplicates(retained: pd.DataFrame) -> pd.DataFrame:
    """One row per standardized parent; keep provenance from all source records."""
    rows = []
    for parent_smi, g in retained.groupby("parent_SMILES", sort=True, dropna=False):
        g = g.sort_values("ID")
        rep = g.iloc[0].copy()
        rep["representative_ID"] = rep["ID"]
        rep["merged_record_count"] = len(g)
        rep["merged_IDs"] = ";".join(str(x) for x in g["ID"].tolist())
        rep["merged_Names"] = " | ".join(str(x) for x in g["Name"].tolist())
        if "PubChem CID" in g.columns:
            rep["merged_PubChem_CIDs"] = ";".join(
                sorted({str(x) for x in g["PubChem CID"].dropna() if str(x) not in {"", "*", "nan"}})
            )
        if "Reference" in g.columns:
            rep["merged_References"] = " | ".join(
                dict.fromkeys(str(x) for x in g["Reference"].dropna())
            )
        rows.append(rep)
    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input umami_small_molecules.csv")
    parser.add_argument("--output-dir", default="data", help="Project data directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.output_dir)
    interim = outdir / "interim"
    processed = outdir / "processed"
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {"ID", "Name", "canonical_SMILES"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    processed_rows = [process_row(row) for _, row in df.iterrows()]
    audit = pd.DataFrame(processed_rows)

    excluded = audit[audit["final_action"] == "EXCLUDE"].copy()
    retained = audit[audit["final_action"].isin(["KEEP", "STANDARDIZE"])].copy()

    # Parent-level duplicate information before merging.
    counts = retained["parent_SMILES"].value_counts()
    retained["parent_duplicate_count"] = retained["parent_SMILES"].map(counts).astype(int)
    retained["parent_duplicate_flag"] = retained["parent_duplicate_count"] > 1

    final = aggregate_duplicates(retained)

    # Add concise final naming fields useful for ML pipelines.
    final["ml_SMILES"] = final["parent_SMILES"]
    final["label"] = 1
    final["label_name"] = "umami"

    audit_path = interim / "umami_standardization_audit.csv"
    excluded_path = interim / "umami_excluded_after_standardization.csv"
    retained_path = interim / "umami_retained_before_dedup.csv"
    dup_path = interim / "umami_parent_duplicate_groups.csv"
    final_path = processed / "umami_positive_final.csv"
    summary_path = interim / "umami_finalization_summary.csv"

    audit.to_csv(audit_path, index=False)
    excluded.to_csv(excluded_path, index=False)
    retained.to_csv(retained_path, index=False)

    dup = retained[retained["parent_duplicate_flag"]].sort_values(["parent_SMILES", "ID"])
    dup.to_csv(dup_path, index=False)
    final.to_csv(final_path, index=False)

    summary = pd.DataFrame([
        {"metric": "input_records", "value": len(df)},
        {"metric": "excluded_records", "value": len(excluded)},
        {"metric": "retained_records_before_dedup", "value": len(retained)},
        {"metric": "records_in_duplicate_parent_groups", "value": len(dup)},
        {"metric": "unique_standardized_parent_molecules_final", "value": len(final)},
        {"metric": "keep_unchanged", "value": int((audit["final_action"] == "KEEP").sum())},
        {"metric": "standardized", "value": int((audit["final_action"] == "STANDARDIZE").sum())},
    ])
    summary.to_csv(summary_path, index=False)

    print("\nUmami dataset finalization complete")
    print("=" * 44)
    for _, r in summary.iterrows():
        print(f"{r['metric']:<40} {r['value']}")
    print("\nExcluded records:")
    if len(excluded):
        print(excluded[["ID", "Name", "exclusion_reason"]].to_string(index=False))
    else:
        print("None")
    print("\nOutputs:")
    for p in [audit_path, excluded_path, retained_path, dup_path, final_path, summary_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()

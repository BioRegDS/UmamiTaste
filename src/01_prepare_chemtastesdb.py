#!/usr/bin/env python3
"""Prepare a low-molecular-weight umami dataset from ChemTastesDB.

Usage:
    python src/01_prepare_chemtastesdb.py \
        --input data/raw/ChemTastesDB_database.xlsx \
        --output-dir data

Outputs:
    data/interim/umami_all_with_flags.csv
    data/interim/umami_excluded_peptides.csv
    data/processed/umami_small_molecules.csv
    data/interim/umami_dataset_summary.csv

Notes:
- Single amino acids are retained.
- Dipeptides and longer peptides are excluded using a conservative combination of
  name-based and structure-based rules.
- Molecular-weight cutoffs are reported but are not used for exclusion by default.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


# Three-letter amino-acid codes commonly seen in peptide names.
AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # Frequently used alternatives / modified residue abbreviations can be added later.
}

# Standard one-letter amino-acid alphabet. Requiring >= 4 characters avoids many
# ordinary short chemical names while still capturing names such as ANPGPVRDLR.
AA1 = set("ACDEFGHIKLMNPQRSTVWY")

# Peptide-like backbone motifs.
# alpha amino-acid residue: N-C(alpha)-C(=O)[O/N]
ALPHA_AA = Chem.MolFromSmarts("[N;X3,X4+]-[C;X4]-[C;X3](=[O;X1])-[O,N]")
# amide/peptide bond: C(=O)-N, excluding obvious sulfonamides etc. by construction.
AMIDE = Chem.MolFromSmarts("[C;X3](=[O;X1])-[N;X3,X4+]")


def canonicalize_smiles(smiles: str) -> tuple[Optional[Chem.Mol], Optional[str], Optional[str]]:
    """Return RDKit mol, canonical SMILES, and error string."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None, None, "missing_smiles"
    s = str(smiles).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None, None, "rdkit_parse_failed"
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return mol, can, None


def murcko_smiles(mol: Chem.Mol) -> str:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def peptide_name_evidence(name: str) -> tuple[bool, str, int]:
    """Detect peptide-like names without classifying single amino acids as peptides."""
    if pd.isna(name):
        return False, "", 0
    raw = str(name).strip()
    upper = raw.upper()

    # Hyphen-separated 3-letter residues: Ala-Glu, Ala-Glu-Ala, etc.
    tokens = [t for t in re.split(r"[-\s]+", upper) if t]
    aa3_hits = sum(t in AA3 for t in tokens)
    if aa3_hits >= 2:
        return True, f"name_3letter_residues:{aa3_hits}", aa3_hits

    # Compact one-letter peptide notation, e.g. ANPGPVRDLR.
    compact = raw.strip()
    if re.fullmatch(r"[A-Z]{4,30}", compact) and all(c in AA1 for c in compact):
        return True, f"name_1letter_sequence:{len(compact)}", len(compact)

    return False, "", aa3_hits


def structure_peptide_evidence(mol: Chem.Mol) -> tuple[bool, str, int, int]:
    """Conservatively flag dipeptide-or-longer structures.

    We require either >=2 alpha-amino-acid-like residues plus >=1 amide bond,
    or >=2 alpha-amino-acid-like residues even if protonation/tautomerism causes
    an amide SMARTS miss. Single amino acids are therefore retained.
    """
    aa_matches = mol.GetSubstructMatches(ALPHA_AA) if ALPHA_AA is not None else ()
    amide_matches = mol.GetSubstructMatches(AMIDE) if AMIDE is not None else ()
    n_aa = len(aa_matches)
    n_amide = len(amide_matches)

    if n_aa >= 2 and n_amide >= 1:
        return True, f"structure_alphaAA:{n_aa};amide:{n_amide}", n_aa, n_amide
    return False, "", n_aa, n_amide


def metal_or_inorganic_flag(mol: Chem.Mol) -> bool:
    """Flag compounds containing metals/metalloids outside common organic elements."""
    common = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}  # H,B,C,N,O,F,Si,P,S,Cl,Br,I
    return any(atom.GetAtomicNum() not in common for atom in mol.GetAtoms())


def analyze_row(row: pd.Series) -> dict:
    mol, can, err = canonicalize_smiles(row.get("canonical SMILES"))
    base = {
        "ID": row.get("ID"),
        "Name": row.get("Name"),
        "PubChem CID": row.get("PubChem CID"),
        "CAS number": row.get("CAS number"),
        "original_SMILES": row.get("canonical SMILES"),
        "canonical_SMILES": can,
        "Taste": row.get("Taste"),
        "Class taste": row.get("Class taste"),
        "Reference": row.get("Reference_(cod)/[pp]"),
        "parse_error": err or "",
    }
    if mol is None:
        base.update({
            "MW": None, "HeavyAtomCount": None, "HBD": None, "HBA": None,
            "RotatableBonds": None, "TPSA": None, "FormalCharge": None,
            "MurckoScaffold": "", "alphaAA_motifs": None, "amide_bonds": None,
            "peptide_flag": False, "peptide_reason": "unparsed",
            "high_MW_500": False, "metal_or_inorganic_flag": False,
        })
        return base

    name_flag, name_reason, _ = peptide_name_evidence(row.get("Name"))
    struct_flag, struct_reason, n_aa, n_amide = structure_peptide_evidence(mol)
    peptide = name_flag or struct_flag
    reasons = ";".join(r for r in [name_reason, struct_reason] if r)

    base.update({
        "MW": round(Descriptors.MolWt(mol), 4),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 4),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "MurckoScaffold": murcko_smiles(mol),
        "alphaAA_motifs": n_aa,
        "amide_bonds": n_amide,
        "peptide_flag": peptide,
        "peptide_reason": reasons,
        "high_MW_500": Descriptors.MolWt(mol) > 500,
        "metal_or_inorganic_flag": metal_or_inorganic_flag(mol),
    })
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ChemTastesDB low-molecular umami dataset")
    parser.add_argument("--input", required=True, help="Path to ChemTastesDB_database.xlsx")
    parser.add_argument("--output-dir", default="data", help="Base output directory (default: data)")
    parser.add_argument("--sheet", default="ChemTastesDB", help="Excel sheet name")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_base = Path(args.output_dir)
    interim = out_base / "interim"
    processed = out_base / "processed"
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path, sheet_name=args.sheet, engine="openpyxl")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()

    required = {"ID", "Name", "canonical SMILES", "Class taste"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    umami = df[df["Class taste"].astype(str).str.strip().str.casefold() == "umaminess"].copy()
    records = pd.DataFrame([analyze_row(row) for _, row in umami.iterrows()])

    # Duplicate canonical SMILES are retained in the audit file but removed from the final dataset.
    records["duplicate_canonical_SMILES"] = records["canonical_SMILES"].duplicated(keep=False) & records["canonical_SMILES"].notna()

    audit_path = interim / "umami_all_with_flags.csv"
    records.to_csv(audit_path, index=False)

    excluded = records[records["peptide_flag"]].copy()
    excluded.to_csv(interim / "umami_excluded_peptides.csv", index=False)

    small = records[(~records["peptide_flag"]) & (records["parse_error"] == "")].copy()
    small = small.drop_duplicates(subset=["canonical_SMILES"], keep="first").reset_index(drop=True)
    small.to_csv(processed / "umami_small_molecules.csv", index=False)

    summary = pd.DataFrame([
        ("all_rows_in_workbook", len(df)),
        ("umami_rows_raw", len(umami)),
        ("umami_parse_success", int((records["parse_error"] == "").sum())),
        ("umami_parse_failed", int((records["parse_error"] != "").sum())),
        ("peptide_flagged", int(records["peptide_flag"].sum())),
        ("nonpeptide_before_dedup", int((~records["peptide_flag"] & (records["parse_error"] == "")).sum())),
        ("duplicate_canonical_smiles_rows", int(records["duplicate_canonical_SMILES"].sum())),
        ("final_small_molecule_rows", len(small)),
        ("final_high_MW_over_500", int(small["high_MW_500"].sum())),
        ("final_metal_or_inorganic", int(small["metal_or_inorganic_flag"].sum())),
    ], columns=["metric", "value"])
    summary.to_csv(interim / "umami_dataset_summary.csv", index=False)

    print("=== ChemTastesDB umami preparation ===")
    for _, r in summary.iterrows():
        print(f"{r['metric']}: {r['value']}")
    print(f"\nAudit:     {audit_path}")
    print(f"Excluded:  {interim / 'umami_excluded_peptides.csv'}")
    print(f"Processed: {processed / 'umami_small_molecules.csv'}")
    print(f"Summary:   {interim / 'umami_dataset_summary.csv'}")


if __name__ == "__main__":
    main()

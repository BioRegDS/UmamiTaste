#!/usr/bin/env python3
"""Prepare standardized negative datasets for low-molecular-weight umami prediction.

Inputs
------
1) ChemTastesDB Excel workbook (raw source)
2) Final standardized umami-positive dataset produced by 02_finalize_umami_dataset.py

Outputs
-------
Primary negative dataset:
  Bitter + Sweet + Sour + Salty + Tasteless

Strict negative dataset:
  Tasteless only

For both datasets the script:
- excludes dipeptides and longer peptides (single amino acids are retained),
- removes/standardizes counterions and salt forms,
- excludes true mixtures containing >1 different substantive molecular species,
- deduplicates at standardized parent-structure level,
- detects standardized parent structures that conflict with the positive umami set,
- removes those conflicts from the model-ready negative datasets,
- writes complete audit trails,
- optionally writes combined positive/negative model-ready CSV files.

Example
-------
python src/03_prepare_negative_dataset.py \
    --chem-tastes data/raw/ChemTastesDB_database.xlsx \
    --positive data/processed/umami_positive_final.csv \
    --output-dir data
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


# -----------------------------
# Dataset definitions
# -----------------------------
PRIMARY_CLASSES = {
    "Bitterness",
    "Sweetness",
    "Sourness",
    "Saltiness",
    "Tastelessness",
}
STRICT_CLASSES = {"Tastelessness"}

# Fragments with <=1 heavy atom are treated as simple counterions/metals.
MIN_SUBSTANTIVE_HEAVY_ATOMS = 2

# -----------------------------
# Peptide detection (same logic as 01_prepare_chemtastesdb.py)
# -----------------------------
AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
AA1 = set("ACDEFGHIKLMNPQRSTVWY")
ALPHA_AA = Chem.MolFromSmarts("[N;X3,X4+]-[C;X4]-[C;X3](=[O;X1])-[O,N]")
AMIDE = Chem.MolFromSmarts("[C;X3](=[O;X1])-[N;X3,X4+]")


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


def peptide_name_evidence(name: object) -> Tuple[bool, str]:
    if pd.isna(name):
        return False, ""
    raw = str(name).strip()
    upper = raw.upper()

    tokens = [t for t in re.split(r"[-\s]+", upper) if t]
    aa3_hits = sum(t in AA3 for t in tokens)
    if aa3_hits >= 2:
        return True, f"name_3letter_residues:{aa3_hits}"

    if re.fullmatch(r"[A-Z]{4,30}", raw) and all(c in AA1 for c in raw):
        return True, f"name_1letter_sequence:{len(raw)}"

    return False, ""


def structure_peptide_evidence(mol: Chem.Mol) -> Tuple[bool, str, int, int]:
    aa_matches = mol.GetSubstructMatches(ALPHA_AA) if ALPHA_AA is not None else ()
    amide_matches = mol.GetSubstructMatches(AMIDE) if AMIDE is not None else ()
    n_aa = len(aa_matches)
    n_amide = len(amide_matches)
    if n_aa >= 2 and n_amide >= 1:
        return True, f"structure_alphaAA:{n_aa};amide:{n_amide}", n_aa, n_amide
    return False, "", n_aa, n_amide


def fragment_signature(mol: Chem.Mol) -> str:
    """Neutralized signature used only to compare substantive fragments."""
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
        if frag.GetNumHeavyAtoms() >= MIN_SUBSTANTIVE_HEAVY_ATOMS:
            substantive.append(frag)
        else:
            counterions.append(frag)
    return substantive, counterions


def standardize_parent(parent: Chem.Mol) -> Chem.Mol:
    """Match the parent standardization used for the positive dataset."""
    m = rdMolStandardize.Cleanup(parent)
    m = rdMolStandardize.FragmentParent(m)
    m = rdMolStandardize.Uncharger().uncharge(m)
    Chem.SanitizeMol(m)
    return m


def describe_parent(mol: Chem.Mol) -> Dict[str, object]:
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


def process_raw_row(row: pd.Series) -> Dict[str, object]:
    """Parse, peptide-filter, mixture-check, and standardize one raw ChemTastesDB row."""
    out: Dict[str, object] = {
        "ID": row.get("ID"),
        "Name": row.get("Name"),
        "PubChem CID": row.get("PubChem CID"),
        "CAS number": row.get("CAS number"),
        "original_SMILES": row.get("canonical SMILES"),
        "Taste": row.get("Taste"),
        "Class taste": row.get("Class taste"),
        "Reference": row.get("Reference_(cod)/[pp]"),
        "parse_error": "",
        "peptide_flag": False,
        "peptide_reason": "",
        "alphaAA_motifs": None,
        "amide_bonds": None,
        "final_action": "",
        "exclusion_reason": "",
        "n_fragments": 0,
        "n_substantive_fragments": 0,
        "n_unique_substantive_parents": 0,
        "counterion_SMILES": "",
        "substantive_fragment_SMILES": "",
        "parent_SMILES": "",
        "standardization_note": "",
    }

    smi = str(row.get("canonical SMILES", "") or "").strip()
    if not smi:
        out["parse_error"] = "missing_smiles"
        out["final_action"] = "EXCLUDE"
        out["exclusion_reason"] = "Missing SMILES"
        return out

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        out["parse_error"] = "rdkit_parse_failed"
        out["final_action"] = "EXCLUDE"
        out["exclusion_reason"] = "SMILES parse failure"
        return out

    out["canonical_SMILES"] = canonical_smiles(mol)

    name_flag, name_reason = peptide_name_evidence(row.get("Name"))
    struct_flag, struct_reason, n_aa, n_amide = structure_peptide_evidence(mol)
    out["alphaAA_motifs"] = n_aa
    out["amide_bonds"] = n_amide
    out["peptide_flag"] = bool(name_flag or struct_flag)
    out["peptide_reason"] = ";".join(x for x in [name_reason, struct_reason] if x)
    if out["peptide_flag"]:
        out["final_action"] = "EXCLUDE"
        out["exclusion_reason"] = "Dipeptide-or-longer peptide"
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

    if len(unique_signatures) > 1:
        out["final_action"] = "EXCLUDE"
        out["exclusion_reason"] = "Multiple different substantive molecular species (mixture)"
        return out

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
    if parent_smi != out.get("canonical_SMILES", ""):
        notes.append("parent representation standardized/uncharged")

    out["standardization_note"] = "; ".join(notes) if notes else "unchanged"
    out["final_action"] = "STANDARDIZE" if notes else "KEEP"
    out.update(describe_parent(parent))
    return out


def aggregate_negative_duplicates(retained: pd.DataFrame) -> pd.DataFrame:
    """Collapse standardized parent duplicates while retaining provenance and source classes."""
    rows = []
    for parent_smi, g in retained.groupby("parent_SMILES", sort=True, dropna=False):
        g = g.sort_values("ID")
        rep = g.iloc[0].copy()
        rep["representative_ID"] = rep["ID"]
        rep["merged_record_count"] = len(g)
        rep["merged_IDs"] = ";".join(str(x) for x in g["ID"].tolist())
        rep["merged_Names"] = " | ".join(str(x) for x in g["Name"].tolist())
        rep["merged_negative_classes"] = ";".join(sorted(set(g["Class taste"].astype(str))))
        rep["merged_Tastes"] = " | ".join(dict.fromkeys(str(x) for x in g["Taste"].dropna()))
        if "PubChem CID" in g.columns:
            rep["merged_PubChem_CIDs"] = ";".join(
                sorted({str(x) for x in g["PubChem CID"].dropna() if str(x) not in {"", "*", "nan"}})
            )
        rep["merged_References"] = " | ".join(
            dict.fromkeys(str(x) for x in g["Reference"].dropna())
        )
        rows.append(rep)
    return pd.DataFrame(rows).reset_index(drop=True)


def prepare_one_negative_set(
    raw: pd.DataFrame,
    allowed_classes: set[str],
    positive_parents: set[str],
    tag: str,
    interim: Path,
    processed: Path,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    subset = raw[raw["Class taste"].isin(allowed_classes)].copy()
    audit = pd.DataFrame([process_raw_row(row) for _, row in subset.iterrows()])

    excluded_pre = audit[audit["final_action"] == "EXCLUDE"].copy()
    retained = audit[audit["final_action"].isin(["KEEP", "STANDARDIZE"])].copy()

    counts = retained["parent_SMILES"].value_counts()
    retained["parent_duplicate_count"] = retained["parent_SMILES"].map(counts).astype(int)
    retained["parent_duplicate_flag"] = retained["parent_duplicate_count"] > 1

    dedup = aggregate_negative_duplicates(retained)
    dedup["positive_structure_conflict"] = dedup["parent_SMILES"].isin(positive_parents)

    conflicts = dedup[dedup["positive_structure_conflict"]].copy()
    final = dedup[~dedup["positive_structure_conflict"]].copy().reset_index(drop=True)
    final["ml_SMILES"] = final["parent_SMILES"]
    final["label"] = 0
    final["label_name"] = "non_umami"
    final["negative_set"] = tag

    # Audit outputs
    audit.to_csv(interim / f"negative_{tag}_standardization_audit.csv", index=False)
    excluded_pre.to_csv(interim / f"negative_{tag}_excluded_before_dedup.csv", index=False)
    retained.to_csv(interim / f"negative_{tag}_retained_before_dedup.csv", index=False)
    retained[retained["parent_duplicate_flag"]].sort_values(["parent_SMILES", "ID"]).to_csv(
        interim / f"negative_{tag}_parent_duplicate_groups.csv", index=False
    )
    conflicts.to_csv(interim / f"negative_{tag}_positive_structure_conflicts.csv", index=False)
    final.to_csv(processed / f"negative_{tag}_final.csv", index=False)

    metrics = {
        "raw_selected_records": len(subset),
        "excluded_before_dedup": len(excluded_pre),
        "retained_before_dedup": len(retained),
        "unique_parent_structures_before_conflict_removal": len(dedup),
        "positive_structure_conflicts": len(conflicts),
        "final_unique_negative_parent_structures": len(final),
        "peptide_excluded": int((audit["peptide_flag"] == True).sum()),
        "parse_failures": int((audit["parse_error"].astype(str) != "").sum()),
        "mixture_excluded": int(
            audit["exclusion_reason"].astype(str).str.contains("mixture", case=False, na=False).sum()
        ),
    }
    return final, metrics


def load_positive(path: Path) -> Tuple[pd.DataFrame, set[str]]:
    pos = pd.read_csv(path)
    col = "parent_SMILES" if "parent_SMILES" in pos.columns else "ml_SMILES"
    if col not in pos.columns:
        raise ValueError("Positive dataset must contain parent_SMILES or ml_SMILES")
    parents = set(pos[col].dropna().astype(str).str.strip())
    return pos, parents


def write_combined_dataset(pos: pd.DataFrame, neg: pd.DataFrame, path: Path, tag: str) -> None:
    """Write a concise model-ready dataset while preserving provenance columns when available."""
    pos2 = pos.copy()
    if "ml_SMILES" not in pos2.columns:
        pos2["ml_SMILES"] = pos2.get("parent_SMILES")
    pos2["label"] = 1
    pos2["label_name"] = "umami"
    pos2["dataset_role"] = "positive"
    pos2["negative_set"] = tag

    neg2 = neg.copy()
    neg2["dataset_role"] = "negative"

    common_core = ["ml_SMILES", "label", "label_name", "dataset_role", "negative_set"]
    extra = [
        "parent_SMILES", "Name", "representative_ID", "ID", "merged_record_count",
        "merged_IDs", "merged_Names", "merged_negative_classes", "Reference",
        "merged_References", "parent_MW", "parent_MurckoScaffold",
    ]
    cols = []
    for c in common_core + extra:
        if c in pos2.columns or c in neg2.columns:
            cols.append(c)

    combined = pd.concat([
        pos2.reindex(columns=cols),
        neg2.reindex(columns=cols),
    ], ignore_index=True)
    combined.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ChemTastesDB negative datasets for umami prediction")
    parser.add_argument("--chem-tastes", required=True, help="Path to ChemTastesDB_database.xlsx")
    parser.add_argument("--positive", required=True, help="Path to umami_positive_final.csv")
    parser.add_argument("--output-dir", default="data", help="Project data directory")
    parser.add_argument("--sheet", default="ChemTastesDB", help="Excel sheet name")
    args = parser.parse_args()

    chem_path = Path(args.chem_tastes)
    pos_path = Path(args.positive)
    outdir = Path(args.output_dir)
    interim = outdir / "interim"
    processed = outdir / "processed"
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(chem_path, sheet_name=args.sheet, engine="openpyxl")
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")].copy()

    required = {"ID", "Name", "canonical SMILES", "Class taste"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required ChemTastesDB columns: {sorted(missing)}")

    pos, positive_parents = load_positive(pos_path)

    primary, primary_metrics = prepare_one_negative_set(
        raw, PRIMARY_CLASSES, positive_parents, "primary", interim, processed
    )
    strict, strict_metrics = prepare_one_negative_set(
        raw, STRICT_CLASSES, positive_parents, "strict", interim, processed
    )

    # Combined model-ready datasets
    write_combined_dataset(pos, primary, processed / "umami_primary_dataset.csv", "primary")
    write_combined_dataset(pos, strict, processed / "umami_strict_dataset.csv", "strict")

    summary_rows = []
    for tag, metrics in [("primary", primary_metrics), ("strict", strict_metrics)]:
        for metric, value in metrics.items():
            summary_rows.append({"dataset": tag, "metric": metric, "value": value})
    summary_rows.extend([
        {"dataset": "positive", "metric": "final_unique_positive_parent_structures", "value": len(positive_parents)},
        {"dataset": "primary_combined", "metric": "total_model_rows", "value": len(pos) + len(primary)},
        {"dataset": "strict_combined", "metric": "total_model_rows", "value": len(pos) + len(strict)},
    ])
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(interim / "negative_dataset_summary.csv", index=False)

    print("=== Negative dataset preparation ===")
    print(f"Positive unique parent structures: {len(positive_parents)}")
    for tag, metrics in [("PRIMARY", primary_metrics), ("STRICT", strict_metrics)]:
        print(f"\n[{tag}]")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    print("\nModel-ready files:")
    print(processed / "negative_primary_final.csv")
    print(processed / "negative_strict_final.csv")
    print(processed / "umami_primary_dataset.csv")
    print(processed / "umami_strict_dataset.csv")
    print("\nSummary:")
    print(interim / "negative_dataset_summary.csv")


if __name__ == "__main__":
    main()

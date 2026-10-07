"""
Curation and Leak-Free Data Splitting Pipeline.
1. Standardizes SMILES and computes canonical InChIKeys using RDKit.
2. Partitions primary Karim dataset into Train (80%), Val (10%), Test (10%).
3. Curates the independent external Doddareddy dataset and removes any overlapping
   molecules (by InChIKey) to ensure 100% leak-free external validation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.cheminformatics import standardize_molecule, validate_and_parse_smiles

RAW_DIR = PROJECT_ROOT / "backend" / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Validates, standardizes, and annotates molecules with canonical SMILES and InChIKey.
    """
    print(f"\n[CURATING] Processing {dataset_name} ({len(df)} initial rows)...")
    valid_rows = []

    for idx, row in df.iterrows():
        raw_smiles = str(row["Drug"]).strip()
        label = int(row["Y"])

        mol, err = validate_and_parse_smiles(raw_smiles)
        if mol is None:
            continue

        try:
            std_mol, can_smiles, inchi_key = standardize_molecule(mol)
            if std_mol.GetNumHeavyAtoms() < 3:
                continue

            valid_rows.append({
                "drug_id": str(row.get("Drug_ID", idx)),
                "raw_smiles": raw_smiles,
                "smiles": can_smiles,
                "inchi_key": inchi_key,
                "y": label
            })
        except Exception:
            continue

    cleaned_df = pd.DataFrame(valid_rows)
    # Deduplicate within dataset by InChIKey
    cleaned_df = cleaned_df.drop_duplicates(subset=["inchi_key"]).reset_index(drop=True)
    print(f"  -> {len(cleaned_df)} valid unique molecules in {dataset_name}")
    print(f"  -> Class distribution: {cleaned_df['y'].value_counts().to_dict()}")
    return cleaned_df


def main():
    karim_path = RAW_DIR / "herg_karim.tsv"
    doddareddy_path = RAW_DIR / "herg_doddareddy.tsv"

    if not karim_path.exists() or not doddareddy_path.exists():
        raise FileNotFoundError("Raw datasets not found. Please run download_data.py first.")

    # 1. Clean primary dataset
    karim_raw = pd.read_csv(karim_path, sep="\t")
    karim_df = clean_dataframe(karim_raw, "Karim et al. (Primary Dataset)")

    # 2. Stratified Train / Val / Test split (80 / 10 / 10)
    train_df, temp_df = train_test_split(
        karim_df,
        test_size=0.20,
        random_state=42,
        stratify=karim_df["y"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["y"]
    )

    print(f"\n[SPLIT] Primary Dataset Split:")
    print(f"  Train: {len(train_df)} molecules ({train_df['y'].sum()} blockers, {len(train_df) - train_df['y'].sum()} non-blockers)")
    print(f"  Val:   {len(val_df)} molecules ({val_df['y'].sum()} blockers, {len(val_df) - val_df['y'].sum()} non-blockers)")
    print(f"  Test:  {len(test_df)} molecules ({test_df['y'].sum()} blockers, {len(test_df) - test_df['y'].sum()} non-blockers)")

    # Save primary splits
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    # 3. Clean and isolate external dataset
    doddareddy_raw = pd.read_csv(doddareddy_path, sep="\t")
    doddareddy_df = clean_dataframe(doddareddy_raw, "Doddareddy et al. (External Benchmark)")

    # Check and eliminate overlap with primary dataset
    primary_inchikeys = set(karim_df["inchi_key"])
    overlapping = doddareddy_df[doddareddy_df["inchi_key"].isin(primary_inchikeys)]
    external_clean = doddareddy_df[~doddareddy_df["inchi_key"].isin(primary_inchikeys)].reset_index(drop=True)

    print(f"\n[LEAKAGE AUDIT] Overlap Analysis:")
    print(f"  Total raw external molecules: {len(doddareddy_df)}")
    print(f"  Molecules found in primary training corpus: {len(overlapping)} (REMOVED to prevent leakage)")
    print(f"  Final strictly independent external validation set: {len(external_clean)} molecules")
    print(f"  External class distribution: {external_clean['y'].value_counts().to_dict()}")

    external_clean.to_csv(PROCESSED_DIR / "external_test.csv", index=False)
    print(f"\n[SUCCESS] All datasets saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

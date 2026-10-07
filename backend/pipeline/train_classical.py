"""
Training Classical Fingerprint-based Machine Learning Models (Random Forest) for hERG Inhibition.
Uses ECFP4 (Morgan radius 2, 1024-bit) representations.
Saves model artifacts and training fingerprints for Applicability Domain queries.
"""

import sys
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.cheminformatics import compute_ecfp4_fingerprint, FP_RADIUS, FP_N_BITS, USE_CHIRALITY
from backend.models.applicability_domain import ApplicabilityDomainEstimator

PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def smiles_to_fps(smiles_list: list) -> np.ndarray:
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fp = np.zeros(FP_N_BITS, dtype=np.int8)
        else:
            fp, _ = compute_ecfp4_fingerprint(mol, radius=FP_RADIUS, n_bits=FP_N_BITS, use_chirality=USE_CHIRALITY)
        fps.append(fp)
    return np.array(fps, dtype=np.uint8)


def main():
    print("=" * 60)
    print("TRAINING CLASSICAL RANDOM FOREST MODEL")
    print("=" * 60)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    ext_df = pd.read_csv(PROCESSED_DIR / "external_test.csv")

    print(f"Loaded datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Ext={len(ext_df)}")

    # Extract ECFP4 fingerprints
    print("Generating ECFP4 (Morgan r=2, 1024 bits) fingerprints...")
    t0 = time.time()
    X_train = smiles_to_fps(train_df["smiles"].tolist())
    y_train = train_df["y"].values

    X_val = smiles_to_fps(val_df["smiles"].tolist())
    y_val = val_df["y"].values

    X_test = smiles_to_fps(test_df["smiles"].tolist())
    y_test = test_df["y"].values

    X_ext = smiles_to_fps(ext_df["smiles"].tolist())
    y_ext = ext_df["y"].values
    print(f"Fingerprints computed in {round(time.time() - t0, 2)}s.")

    # Save training fingerprints for Applicability Domain queries
    print("Saving training fingerprints for Applicability Domain estimation...")
    ad_estimator = ApplicabilityDomainEstimator()
    ad_estimator.fit(X_train)
    ad_estimator.save(str(ARTIFACTS_DIR / "training_fingerprints.npy"))

    # Train Random Forest Classifier
    print("Training Random Forest Classifier (300 estimators, balanced subsample)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)
    print(f"Model trained in {train_time}s.")

    # Save trained RF model
    model_path = ARTIFACTS_DIR / "rf_model.joblib"
    joblib.dump(rf, model_path, compress=3)
    print(f"Model saved to {model_path} ({model_path.stat().st_size} bytes)")

    # Save preprocessing metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 300,
        "max_depth": 30,
        "fingerprint": {
            "type": "Morgan / ECFP4",
            "radius": FP_RADIUS,
            "n_bits": FP_N_BITS,
            "use_chirality": USE_CHIRALITY
        },
        "training_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "internal_test_samples": int(len(test_df)),
        "external_test_samples": int(len(ext_df)),
        "random_state": 42
    }
    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Classical model training complete.")


if __name__ == "__main__":
    main()

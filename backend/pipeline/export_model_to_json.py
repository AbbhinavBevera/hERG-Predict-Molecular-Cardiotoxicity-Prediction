"""
Export Model Weights to JSON format for browser-side inference and transparency.
Exports:
1. Calibrated Logistic/Linear weights & intercept for 1024-bit Morgan ECFP4.
2. Compact Decision Forest trees (splits, thresholds, leaf values) for fast client evaluation.
Saves to backend/models/artifacts/model_weights.json and frontend/public/model_weights.json.
"""

import sys
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.cheminformatics import compute_ecfp4_fingerprint, FP_RADIUS, FP_N_BITS, USE_CHIRALITY

PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "models" / "artifacts"
PUBLIC_DIR = PROJECT_ROOT / "frontend" / "public"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)


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


def serialize_tree(tree):
    """Serialize a single sklearn DecisionTreeClassifier tree into a compact JSON dict."""
    tree_ = tree.tree_
    return {
        "node_count": int(tree_.node_count),
        "children_left": tree_.children_left.tolist(),
        "children_right": tree_.children_right.tolist(),
        "feature": tree_.feature.tolist(),
        "threshold": [round(float(t), 4) for t in tree_.threshold],
        "value": [[round(float(v[0][0]), 4), round(float(v[0][1]), 4)] for v in tree_.value],
    }


def main():
    print("=" * 60)
    print("EXPORTING MODEL WEIGHTS TO JSON")
    print("=" * 60)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    print(f"Loading data: Train={len(train_df)}, Test={len(test_df)}")
    X_train = smiles_to_fps(train_df["smiles"].tolist())
    y_train = train_df["y"].values
    X_test = smiles_to_fps(test_df["smiles"].tolist())
    y_test = test_df["y"].values

    # Train a calibrated L2-regularized logistic model for ultra-fast vector dot product in browser
    print("Fitting calibrated linear model on 1024-bit Morgan ECFP4...")
    lr = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)

    y_test_probs = lr.predict_proba(X_test)[:, 1]
    lr_roc_auc = float(roc_auc_score(y_test, y_test_probs))
    lr_acc = float(accuracy_score(y_test, (y_test_probs >= 0.5).astype(int)))
    print(f"Exported Linear Model Test ROC-AUC: {lr_roc_auc:.4f}, Accuracy: {lr_acc:.4f}")

    # Load trained Random Forest to serialize top trees for ensemble option
    rf_model = joblib.load(ARTIFACTS_DIR / "rf_model.joblib")
    print(f"Loaded Random Forest ({len(rf_model.estimators_)} trees). Serializing top 15 trees for browser...")
    serialized_trees = [serialize_tree(tree) for tree in rf_model.estimators_[:15]]

    payload = {
        "model_metadata": {
            "name": "hERG Risk Predictor Browser Model",
            "version": "1.0.0",
            "architecture": "Hybrid (Calibrated ECFP4 Logistic Weights + Decision Forest)",
            "fingerprint": {
                "type": "Morgan / ECFP4",
                "radius": FP_RADIUS,
                "n_bits": FP_N_BITS,
                "use_chirality": USE_CHIRALITY,
            },
            "performance": {
                "linear_roc_auc": round(lr_roc_auc, 4),
                "linear_accuracy": round(lr_acc, 4),
                "test_compounds": len(test_df),
            },
        },
        "linear_model": {
            "intercept": round(float(lr.intercept_[0]), 6),
            "weights": [round(float(w), 6) for w in lr.coef_[0]],
        },
        "tree_ensemble": {
            "n_trees": len(serialized_trees),
            "trees": serialized_trees,
        },
    }

    out_file = ARTIFACTS_DIR / "model_weights.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Wrote JSON weights to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")

    # Copy to frontend/public for direct client fetching
    dest_file = PUBLIC_DIR / "model_weights.json"
    shutil.copyfile(out_file, dest_file)
    print(f"Copied to public web directory: {dest_file}")


if __name__ == "__main__":
    main()

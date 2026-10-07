"""
Comprehensive Scientific Evaluation Script.
Evaluates both Classical ECFP4 Random Forest and PyTorch Molecular GNN on:
1. Internal Test Set (Karim et al. test split, n=1309)
2. External Validation Set (Doddareddy et al. leak-free independent set, n=182)

Measures and records:
- ROC-AUC, PR-AUC, Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1, MCC
- Complete ROC and PR curves for frontend rendering
- Confusion matrices
Saves all verified metrics to backend/models/artifacts/benchmark_metrics.json.
"""

import sys
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.cheminformatics import compute_ecfp4_fingerprint, FP_RADIUS, FP_N_BITS, USE_CHIRALITY
from backend.models.gnn import MolecularGNN, mol_to_graph_data, collate_graph_batch, NODE_FEATURE_DIM

PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "models" / "artifacts"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Curve points sampled for clean JSON / charts
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    rec, prec, _ = precision_recall_curve(y_true, y_prob)

    # Subsample curve points for smooth charting without bloating JSON
    step_roc = max(1, len(fpr) // 40)
    step_pr = max(1, len(rec) // 40)

    roc_pts = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr[::step_roc], tpr[::step_roc])]
    if roc_pts[-1] != {"fpr": 1.0, "tpr": 1.0}:
        roc_pts.append({"fpr": 1.0, "tpr": 1.0})

    pr_pts = [{"recall": round(float(r), 4), "precision": round(float(p), 4)} for r, p in zip(rec[::step_pr], prec[::step_pr])]

    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp) if (tn + fp) > 0 else 0), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        },
        "roc_curve": roc_pts,
        "pr_curve": pr_pts
    }


def predict_rf(rf, smiles_list: list) -> np.ndarray:
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(np.zeros(FP_N_BITS, dtype=np.int8))
        else:
            fp, _ = compute_ecfp4_fingerprint(mol, radius=FP_RADIUS, n_bits=FP_N_BITS, use_chirality=USE_CHIRALITY)
            fps.append(fp)
    X = np.array(fps, dtype=np.uint8)
    return rf.predict_proba(X)[:, 1]


def predict_gnn(gnn, smiles_list: list) -> np.ndarray:
    gnn.eval()
    probs = []
    with torch.no_grad():
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                probs.append(0.5)
                continue
            prob, _ = gnn.predict_molecule(mol)
            probs.append(prob)
    return np.array(probs)


def main():
    print("=" * 60)
    print("EVALUATING ALL MODELS (INTERNAL TEST & EXTERNAL VALIDATION)")
    print("=" * 60)

    # Load datasets
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    ext_df = pd.read_csv(PROCESSED_DIR / "external_test.csv")

    y_test = test_df["y"].values
    y_ext = ext_df["y"].values

    # 1. Load RF model
    rf_path = ARTIFACTS_DIR / "rf_model.joblib"
    if not rf_path.exists():
        raise FileNotFoundError(f"RF model not found at {rf_path}. Run train_classical.py first.")
    rf_model = joblib.load(rf_path)
    print("Loaded Random Forest model.")

    # 2. Load GNN model
    gnn_path = ARTIFACTS_DIR / "gnn_model.pt"
    gnn_model = MolecularGNN(in_dim=NODE_FEATURE_DIM, hidden_dim=96, num_layers=3, dropout=0.20)
    if gnn_path.exists():
        gnn_model.load_state_dict(torch.load(gnn_path, map_location="cpu"))
        print("Loaded PyTorch Molecular GNN model.")
    else:
        print("[WARNING] GNN checkpoint not found. Evaluating RF only.")
        gnn_model = None

    # Predict RF
    rf_test_probs = predict_rf(rf_model, test_df["smiles"].tolist())
    rf_ext_probs = predict_rf(rf_model, ext_df["smiles"].tolist())

    # Metrics for RF
    rf_internal_metrics = compute_metrics(y_test, rf_test_probs)
    rf_external_metrics = compute_metrics(y_ext, rf_ext_probs)

    print("\n--- RANDOM FOREST (ECFP4) RESULTS ---")
    print(f"Internal Test   (n={len(test_df)}): ROC-AUC = {rf_internal_metrics['roc_auc']:.4f} | PR-AUC = {rf_internal_metrics['pr_auc']:.4f} | Acc = {rf_internal_metrics['accuracy']:.4f} | F1 = {rf_internal_metrics['f1']:.4f} | MCC = {rf_internal_metrics['mcc']:.4f}")
    print(f"External Validation (n={len(ext_df)}): ROC-AUC = {rf_external_metrics['roc_auc']:.4f} | PR-AUC = {rf_external_metrics['pr_auc']:.4f} | Acc = {rf_external_metrics['accuracy']:.4f} | F1 = {rf_external_metrics['f1']:.4f} | MCC = {rf_external_metrics['mcc']:.4f}")

    # Predict GNN if available
    gnn_internal_metrics = None
    gnn_external_metrics = None
    if gnn_model is not None:
        gnn_test_probs = predict_gnn(gnn_model, test_df["smiles"].tolist())
        gnn_ext_probs = predict_gnn(gnn_model, ext_df["smiles"].tolist())
        gnn_internal_metrics = compute_metrics(y_test, gnn_test_probs)
        gnn_external_metrics = compute_metrics(y_ext, gnn_ext_probs)

        print("\n--- GRAPH NEURAL NETWORK (GNN) RESULTS ---")
        print(f"Internal Test   (n={len(test_df)}): ROC-AUC = {gnn_internal_metrics['roc_auc']:.4f} | PR-AUC = {gnn_internal_metrics['pr_auc']:.4f} | Acc = {gnn_internal_metrics['accuracy']:.4f} | F1 = {gnn_internal_metrics['f1']:.4f} | MCC = {gnn_internal_metrics['mcc']:.4f}")
        print(f"External Validation (n={len(ext_df)}): ROC-AUC = {gnn_external_metrics['roc_auc']:.4f} | PR-AUC = {gnn_external_metrics['pr_auc']:.4f} | Acc = {gnn_external_metrics['accuracy']:.4f} | F1 = {gnn_external_metrics['f1']:.4f} | MCC = {gnn_external_metrics['mcc']:.4f}")

    benchmark_data = {
        "dataset_metadata": {
            "internal_test_size": int(len(test_df)),
            "internal_test_source": "Karim et al. (2021) / TDC ChEMBL Curated Test Split",
            "external_test_size": int(len(ext_df)),
            "external_test_source": "Doddareddy et al. (2010) / TDC Tox.herg (Strictly Deduplicated, 0 Leakage)",
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "models": {
            "random_forest": {
                "name": "Random Forest (ECFP4)",
                "description": "300-tree ensemble trained on 1024-bit Morgan fingerprints (radius 2)",
                "internal_test": rf_internal_metrics,
                "external_validation": rf_external_metrics
            }
        }
    }

    if gnn_internal_metrics and gnn_external_metrics:
        benchmark_data["models"]["molecular_gnn"] = {
            "name": "Molecular Graph Neural Network (GNN)",
            "description": "3-layer Graph Convolution Network with dual mean+max pooling on atom feature graphs",
            "internal_test": gnn_internal_metrics,
            "external_validation": gnn_external_metrics
        }

    # Save benchmark metrics to JSON artifact
    out_file = ARTIFACTS_DIR / "benchmark_metrics.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n[SUCCESS] Benchmark metrics exported to {out_file}")


if __name__ == "__main__":
    main()

"""
Applicability Domain (AD) Assessment for hERG Prediction.
Implements OECD Principle 3: "A defined domain of applicability".
Uses Tanimoto similarity against the training chemical library to determine whether a query compound
falls within the reliable chemical space of the models.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path


# Default similarity thresholds established on the training distribution
IN_DOMAIN_THRESHOLD = 0.40
BORDERLINE_THRESHOLD = 0.28


class ApplicabilityDomainEstimator:
    def __init__(self, training_fps_path: Optional[str] = None):
        self.training_fps: Optional[np.ndarray] = None  # shape: (N, 1024)
        if training_fps_path and Path(training_fps_path).exists():
            self.load_training_fingerprints(training_fps_path)

    def fit(self, training_fps: np.ndarray):
        """
        Stores the training fingerprints for fast vectorized similarity queries.
        training_fps is a binary 2D array of shape (N, n_bits).
        """
        self.training_fps = np.asarray(training_fps, dtype=np.uint8)

    def save(self, filepath: str):
        np.save(filepath, self.training_fps)

    def load_training_fingerprints(self, filepath: str):
        self.training_fps = np.load(filepath)

    def evaluate(self, query_fp: np.ndarray, k: int = 5) -> Dict[str, Any]:
        """
        Evaluates the applicability domain of a query molecule's fingerprint.
        Calculates:
        - max_tanimoto: Highest Tanimoto similarity to any training molecule
        - knn_tanimoto: Mean Tanimoto similarity to the k nearest neighbors
        - status: 'In-Domain' | 'Borderline' | 'Out-of-Domain'
        - warning: Warning message if outside or borderline
        """
        if self.training_fps is None or len(self.training_fps) == 0:
            return {
                "status": "Unknown",
                "max_tanimoto": 0.0,
                "knn_tanimoto": 0.0,
                "is_reliable": True,
                "warning": "Training fingerprints not loaded. Applicability domain check unavailable."
            }

        q = np.asarray(query_fp, dtype=np.uint8)
        # Vectorized Tanimoto calculation:
        # T(a, b) = dot(a, b) / (sum(a) + sum(b) - dot(a, b))
        q_sum = int(np.sum(q))
        if q_sum == 0:
            return {
                "status": "Out-of-Domain",
                "max_tanimoto": 0.0,
                "knn_tanimoto": 0.0,
                "is_reliable": False,
                "warning": "Molecule produced an empty fingerprint. Highly unusual or ultra-small structure."
            }

        intersection = np.dot(self.training_fps, q)
        train_sums = np.sum(self.training_fps, axis=1)
        union = train_sums + q_sum - intersection
        union = np.maximum(union, 1)  # avoid div by zero

        similarities = intersection / union
        max_tanimoto = float(np.max(similarities))

        # Top k similarities
        k_val = min(k, len(similarities))
        top_k = np.partition(similarities, -k_val)[-k_val:]
        knn_tanimoto = float(np.mean(top_k))

        if max_tanimoto >= IN_DOMAIN_THRESHOLD:
            status = "In-Domain"
            is_reliable = True
            warning = None
            explanation = "Molecule shares high structural similarity with compounds in the training set. Prediction is within the validated domain."
        elif max_tanimoto >= BORDERLINE_THRESHOLD:
            status = "Borderline"
            is_reliable = True
            warning = "Molecule exhibits moderate structural similarity to the training set. Predictions should be interpreted with caution."
            explanation = "Borderline chemical similarity. Key pharmacophore features may be present, but some structural motifs fall into sparse regions of training space."
        else:
            status = "Out-of-Domain"
            is_reliable = False
            warning = "Molecule is OUT-OF-DOMAIN. Chemical space is distant from training compounds. Model prediction is an extrapolation and may be unreliable."
            explanation = "Novel or under-represented chemical scaffold. Highest training Tanimoto similarity is below threshold. Do not rely solely on this computational prediction."

        return {
            "status": status,
            "max_tanimoto": round(max_tanimoto, 4),
            "knn_tanimoto": round(knn_tanimoto, 4),
            "is_reliable": is_reliable,
            "warning": warning,
            "explanation": explanation,
            "thresholds": {
                "in_domain": IN_DOMAIN_THRESHOLD,
                "borderline": BORDERLINE_THRESHOLD
            }
        }

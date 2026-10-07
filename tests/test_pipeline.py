"""
Automated Test Suite for hERG Molecular Risk Predictor.
Tests:
- SMILES validation & standardization
- Fingerprint generation & bitInfo extraction
- Model inference (Random Forest & GNN)
- Applicability domain estimation
- Explainability substructure attribution
- API endpoints
"""

import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.cheminformatics import (
    validate_and_parse_smiles,
    standardize_molecule,
    calculate_descriptors,
    compute_ecfp4_fingerprint,
    render_molecule_svg
)
from backend.models.gnn import MolecularGNN, mol_to_graph_data
from backend.models.applicability_domain import ApplicabilityDomainEstimator
from backend.app.main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_smiles_validation_valid():
    mol, err = validate_and_parse_smiles("c1ccccc1")
    assert mol is not None
    assert err is None


def test_smiles_validation_invalid():
    mol, err = validate_and_parse_smiles("INVALID_C_VALENCE_TEST")
    assert mol is None
    assert err is not None
    assert "could not parse" in err


def test_standardize_and_desalt():
    # Sodium benzoate (desalts to benzoate organic fragment)
    mol, _ = validate_and_parse_smiles("c1ccccc1C(=O)[O-].[Na+]")
    assert mol is not None
    std_mol, can_smiles, inchi_key = standardize_molecule(mol)
    assert "[Na+]" not in can_smiles
    assert len(inchi_key) == 27


def test_fingerprint_generation():
    mol, _ = validate_and_parse_smiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    fp_vec, bit_info = compute_ecfp4_fingerprint(mol, return_bit_info=True)
    assert len(fp_vec) == 1024
    assert np.sum(fp_vec) > 0
    assert bit_info is not None
    assert len(bit_info) > 0


def test_descriptors_calculation():
    mol, _ = validate_and_parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    desc = calculate_descriptors(mol)
    assert desc["molecular_weight"] > 170
    assert desc["formula"] == "C9H8O4"
    assert "logp" in desc
    assert "tpsa" in desc


def test_render_svg():
    mol, _ = validate_and_parse_smiles("c1ccccc1")
    svg = render_molecule_svg(mol, highlight_atoms=[0, 1])
    assert "<svg" in svg
    assert "</svg>" in svg


def test_gnn_graph_conversion():
    mol, _ = validate_and_parse_smiles("c1ccccc1")
    g = mol_to_graph_data(mol)
    assert g is not None
    assert g["x"].size(0) == 6  # 6 benzene carbons
    assert g["edge_index"].size(0) == 2


def test_applicability_domain():
    ad = ApplicabilityDomainEstimator()
    dummy_train = np.zeros((10, 1024), dtype=np.uint8)
    dummy_train[:, 0] = 1
    dummy_train[:, 1] = 1
    ad.fit(dummy_train)

    query = np.zeros(1024, dtype=np.uint8)
    query[0] = 1
    query[1] = 1

    res = ad.evaluate(query)
    assert res["status"] == "In-Domain"
    assert res["max_tanimoto"] == 1.0


def test_api_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["models_ready"] is True


def test_api_predict_endpoint():
    resp = client.post("/api/predict", json={
        "smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c4ccccc4)CC2)cc1",
        "model_choice": "consensus"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert data["prediction"]["is_blocker"] is True
    assert data["prediction"]["probability"] > 0.70
    assert "applicability_domain" in data
    assert "explainability" in data

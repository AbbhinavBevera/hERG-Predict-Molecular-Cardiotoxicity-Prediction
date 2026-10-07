"""
FastAPI Backend Application for hERG Molecular Risk Predictor.
Exposes endpoints for:
- Single molecule prediction with explainability and applicability domain assessment
- Interactive SVG rendering with dynamic atom/bond highlighting
- Batch prediction with validation error auditing and CSV export
- Benchmarks and model comparison metrics
- Reference example molecules
"""

import io
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import torch
from rdkit import Chem

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.cheminformatics import (
    validate_and_parse_smiles,
    standardize_molecule,
    calculate_descriptors,
    compute_ecfp4_fingerprint,
    render_molecule_svg,
    FP_RADIUS,
    FP_N_BITS,
    USE_CHIRALITY
)
from backend.models.gnn import MolecularGNN, NODE_FEATURE_DIM
from backend.models.applicability_domain import ApplicabilityDomainEstimator
from backend.models.explainer import explain_prediction

ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "models" / "artifacts"

# Initialize FastAPI App
app = FastAPI(
    title="hERG Molecular Risk Predictor API",
    description="Computational drug-safety AI for hERG channel cardiotoxicity risk assessment.",
    version="1.0.0"
)

# Enable CORS for local Vite development & production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
models_loaded = False
rf_model = None
gnn_model = None
ad_estimator = None
benchmark_metrics_cache = None
metadata_cache = None


def load_artifacts():
    global models_loaded, rf_model, gnn_model, ad_estimator, benchmark_metrics_cache, metadata_cache
    if models_loaded:
        return

    rf_path = ARTIFACTS_DIR / "rf_model.joblib"
    if rf_path.exists():
        rf_model = joblib.load(rf_path)
        print("[INIT] Loaded Random Forest model.")

    gnn_path = ARTIFACTS_DIR / "gnn_model.pt"
    if gnn_path.exists():
        gnn_model = MolecularGNN(in_dim=NODE_FEATURE_DIM, hidden_dim=96, num_layers=3, dropout=0.20)
        gnn_model.load_state_dict(torch.load(gnn_path, map_location="cpu"))
        gnn_model.eval()
        print("[INIT] Loaded GNN model.")

    fps_path = ARTIFACTS_DIR / "training_fingerprints.npy"
    if fps_path.exists():
        ad_estimator = ApplicabilityDomainEstimator(str(fps_path))
        print(f"[INIT] Loaded Applicability Domain training fingerprints ({len(ad_estimator.training_fps)} compounds).")

    metrics_path = ARTIFACTS_DIR / "benchmark_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            benchmark_metrics_cache = json.load(f)

    meta_path = ARTIFACTS_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata_cache = json.load(f)

    models_loaded = True


@app.on_event("startup")
def startup_event():
    load_artifacts()


# Curated example molecules representing well-established pharmacological benchmarks
REFERENCE_MOLECULES = [
    {
        "name": "Terfenadine",
        "category": "Antihistamine (Known Potent Blocker)",
        "clinical_context": "Withdrawn worldwide after causing fatal torsades de pointes ventricular arrhythmias via potent hERG blockade (IC50 ~ 56 nM).",
        "known_risk": "High Risk (Potent Blocker)",
        "smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c4ccccc4)CC2)cc1"
    },
    {
        "name": "Astemizole",
        "category": "Antihistamine (Known Potent Blocker)",
        "clinical_context": "Second-generation H1 antagonist withdrawn due to severe cardiac QT prolongation and hERG pore binding (IC50 ~ 1 nM).",
        "known_risk": "High Risk (Potent Blocker)",
        "smiles": "COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc5ccc(F)cc5)CC2)cc1"
    },
    {
        "name": "Cisapride",
        "category": "Gastroprokinetic (Known Blocker)",
        "clinical_context": "5-HT4 receptor agonist withdrawn due to high affinity for the hERG channel leading to syncope and sudden cardiac death.",
        "known_risk": "High Risk (Potent Blocker)",
        "smiles": "COc1cc(Cl)c(N)cc1C(=O)NC2CCN(CCCOC3ccc(F)cc3)C(OC)C2"
    },
    {
        "name": "Haloperidol",
        "category": "Antipsychotic (Known Blocker)",
        "clinical_context": "First-generation typical butyrophenone antipsychotic with known concentration-dependent hERG blockage and QT prolongation warnings.",
        "known_risk": "High Risk (Blocker)",
        "smiles": "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c3ccc(Cl)cc3"
    },
    {
        "name": "Dofetilide",
        "category": "Class III Antiarrhythmic (Reference Blocker)",
        "clinical_context": "High-affinity selective IKr/hERG blocker used therapeutically for atrial fibrillation, requires in-hospital ECG monitoring.",
        "known_risk": "High Risk (Potent Blocker)",
        "smiles": "CN(CCc1ccc(NS(=O)(=O)C)cc1)CCc2ccc(NS(=O)(=O)C)cc2"
    },
    {
        "name": "Aspirin",
        "category": "Analgesic / NSAID (Non-Blocker)",
        "clinical_context": "Nonsteroidal anti-inflammatory drug. Very low lipophilicity, devoid of basic amines, negative for hERG blockade.",
        "known_risk": "Low Risk (Non-Blocker)",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O"
    },
    {
        "name": "Caffeine",
        "category": "CNS Stimulant (Non-Blocker)",
        "clinical_context": "Methylxanthine adenosine receptor antagonist. Devoid of hERG channel blockade liability across therapeutic doses.",
        "known_risk": "Low Risk (Non-Blocker)",
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    },
    {
        "name": "Amoxicillin",
        "category": "Beta-Lactam Antibiotic (Non-Blocker)",
        "clinical_context": "Broad-spectrum hydrophilic aminopenicillin, zero hERG affinity, safe cardiac profile.",
        "known_risk": "Low Risk (Non-Blocker)",
        "smiles": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"
    },
    {
        "name": "Metformin",
        "category": "Antidiabetic (Non-Blocker)",
        "clinical_context": "Biguanide first-line type 2 diabetes agent with no hERG channel inhibition.",
        "known_risk": "Low Risk (Non-Blocker)",
        "smiles": "CN(C)C(=N)NC(=N)N"
    },
    {
        "name": "Ibuprofen",
        "category": "Analgesic / NSAID (Non-Blocker)",
        "clinical_context": "Propionic acid NSAID. Lacks basic amine pharmacophore, negative for hERG blockade.",
        "known_risk": "Low Risk (Non-Blocker)",
        "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
    }
]


class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string to predict")
    model_choice: str = Field("consensus", description="Model choice: 'random_forest', 'gnn', or 'consensus'")
    dark_mode: bool = Field(True, description="Whether to render SVG in dark mode")


class HighlightRequest(BaseModel):
    smiles: str
    highlight_atoms: List[int] = Field(default_factory=list)
    dark_mode: bool = True
    width: int = 450
    height: int = 320


class BatchRowResult(BaseModel):
    row_index: int
    raw_smiles: str
    is_valid: bool
    error_message: Optional[str] = None
    canonical_smiles: Optional[str] = None
    prediction_class: Optional[str] = None
    probability: Optional[float] = None
    applicability_domain: Optional[str] = None
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None


@app.get("/api/health")
def health_check():
    load_artifacts()
    return {
        "status": "healthy",
        "models_ready": rf_model is not None and gnn_model is not None,
        "ad_ready": ad_estimator is not None,
        "disclaimer": "This tool provides computational predictions for research and educational purposes. It is not a clinical diagnostic tool and should not be used as the sole basis for drug-safety decisions."
    }


@app.get("/api/examples")
def get_examples():
    return REFERENCE_MOLECULES


@app.get("/api/metrics")
def get_metrics():
    if benchmark_metrics_cache is not None:
        return benchmark_metrics_cache
    metrics_path = ARTIFACTS_DIR / "benchmark_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Benchmark metrics artifact not found.")


@app.get("/api/metadata")
def get_metadata():
    if metadata_cache is not None:
        return metadata_cache
    meta_path = ARTIFACTS_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Model metadata artifact not found.")


@app.post("/api/render-svg")
def render_custom_svg(req: HighlightRequest):
    mol, err = validate_and_parse_smiles(req.smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {err}")
    std_mol, _, _ = standardize_molecule(mol)
    svg = render_molecule_svg(
        std_mol,
        highlight_atoms=req.highlight_atoms,
        width=req.width,
        height=req.height,
        dark_mode=req.dark_mode
    )
    return {"svg": svg}


@app.post("/api/predict")
def predict_single(req: PredictRequest):
    load_artifacts()

    # 1. Validate SMILES
    mol, err = validate_and_parse_smiles(req.smiles)
    if mol is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "SMILES Validation Error",
                "message": err,
                "input_smiles": req.smiles
            }
        )

    # 2. Standardize molecule
    std_mol, canonical_smiles, inchi_key = standardize_molecule(mol)

    # 3. Physicochemical Descriptors
    descriptors = calculate_descriptors(std_mol)

    # 4. Fingerprint computation
    fp_vec, bit_info = compute_ecfp4_fingerprint(std_mol, return_bit_info=True)

    # 5. Model Predictions
    # Classical RF prediction
    rf_prob = 0.5
    if rf_model is not None:
        rf_prob = float(rf_model.predict_proba(fp_vec.reshape(1, -1))[0, 1])

    # GNN prediction
    gnn_prob = 0.5
    if gnn_model is not None:
        gnn_prob, _ = gnn_model.predict_molecule(std_mol)

    # Model choice routing
    if req.model_choice == "random_forest":
        final_prob = rf_prob
        selected_model = "Random Forest (ECFP4)"
    elif req.model_choice == "gnn":
        final_prob = gnn_prob
        selected_model = "Molecular Graph Neural Network (GNN)"
    else:  # consensus
        final_prob = (rf_prob + gnn_prob) / 2.0
        selected_model = "Consensus Ensemble (RF + GNN)"

    is_blocker = final_prob >= 0.5
    prediction_class = "High Risk (hERG Blocker)" if is_blocker else "Low Risk (Non-Blocker)"

    # Risk level categorization
    if final_prob >= 0.75:
        risk_tier = "Very High Cardiotoxicity Risk"
    elif final_prob >= 0.50:
        risk_tier = "Moderate Cardiotoxicity Risk"
    elif final_prob >= 0.25:
        risk_tier = "Low Cardiotoxicity Risk"
    else:
        risk_tier = "Minimal Cardiotoxicity Risk"

    # Confidence metric (|prob - 0.5| * 2)
    confidence_score = round(abs(final_prob - 0.5) * 200, 1)

    # 6. Applicability Domain Check
    ad_result = {}
    if ad_estimator is not None:
        ad_result = ad_estimator.evaluate(fp_vec)
    else:
        ad_result = {"status": "Unknown", "is_reliable": True, "warning": None}

    # 7. Explainability / Substructure attribution
    top_fragments = []
    if rf_model is not None:
        top_fragments = explain_prediction(std_mol, rf_model, rf_prob, top_n=6)

    # 8. Render Default Structure SVG
    # Collect all atoms from top risk-increasing fragments to provide initial informative highlight
    initial_highlight_atoms = []
    for frag in top_fragments:
        if frag.get("direction") == "increases_risk" and frag.get("atom_indices"):
            initial_highlight_atoms.extend(frag["atom_indices"])
            break  # highlight the #1 risk feature by default

    svg = render_molecule_svg(
        std_mol,
        highlight_atoms=initial_highlight_atoms,
        dark_mode=req.dark_mode
    )

    return {
        "smiles": req.smiles,
        "canonical_smiles": canonical_smiles,
        "inchi_key": inchi_key,
        "prediction": {
            "prediction_class": prediction_class,
            "is_blocker": is_blocker,
            "probability": round(final_prob, 4),
            "percentage": round(final_prob * 100, 1),
            "confidence_percentage": confidence_score,
            "risk_tier": risk_tier,
            "model_used": selected_model,
            "model_breakdown": {
                "random_forest_probability": round(rf_prob, 4),
                "gnn_probability": round(gnn_prob, 4)
            }
        },
        "applicability_domain": ad_result,
        "explainability": {
            "top_fragments": top_fragments,
            "pharmacophore_summary": "hERG channel blockade is commonly mediated by basic amine centers and lipophilic aromatic rings lodging within the pore central cavity formed by Tyr652 and Phe656 residues."
        },
        "descriptors": descriptors,
        "svg": svg,
        "disclaimer": "This tool provides computational predictions for research and educational purposes. It is not a clinical diagnostic tool and should not be used as the sole basis for drug-safety decisions."
    }


@app.post("/api/batch")
async def batch_predict(
    file: Optional[UploadFile] = File(None),
    model_choice: str = Query("consensus")
):
    load_artifacts()

    if file is None:
        raise HTTPException(status_code=400, detail="No CSV file uploaded.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    # Locate SMILES column (case-insensitive search)
    smiles_col = None
    for col in df.columns:
        if col.strip().lower() in ["smiles", "smile", "drug", "molecule", "structure", "compound"]:
            smiles_col = col
            break

    if smiles_col is None:
        # Default to first column if no named column matched
        smiles_col = df.columns[0]

    results = []
    total_rows = len(df)
    valid_count = 0
    invalid_count = 0

    for idx, row in df.iterrows():
        raw_val = str(row[smiles_col]).strip() if pd.notna(row[smiles_col]) else ""
        if not raw_val or raw_val.lower() == "nan":
            results.append({
                "row_index": idx + 1,
                "raw_smiles": raw_val,
                "is_valid": False,
                "error_message": "Empty or NaN SMILES cell."
            })
            invalid_count += 1
            continue

        mol, err = validate_and_parse_smiles(raw_val)
        if mol is None:
            results.append({
                "row_index": idx + 1,
                "raw_smiles": raw_val,
                "is_valid": False,
                "error_message": err
            })
            invalid_count += 1
            continue

        try:
            std_mol, can_smiles, _ = standardize_molecule(mol)
            desc = calculate_descriptors(std_mol)
            fp_vec, _ = compute_ecfp4_fingerprint(std_mol)

            # Predictions
            rf_p = float(rf_model.predict_proba(fp_vec.reshape(1, -1))[0, 1]) if rf_model else 0.5
            gnn_p = 0.5
            if gnn_model:
                gnn_p, _ = gnn_model.predict_molecule(std_mol)

            if model_choice == "random_forest":
                prob = rf_p
            elif model_choice == "gnn":
                prob = gnn_p
            else:
                prob = (rf_p + gnn_p) / 2.0

            pred_class = "High Risk (Blocker)" if prob >= 0.5 else "Low Risk (Non-Blocker)"

            # Applicability domain
            ad_status = "In-Domain"
            max_tanimoto = 0.0
            if ad_estimator:
                ad = ad_estimator.evaluate(fp_vec)
                ad_status = ad["status"]
                max_tanimoto = ad["max_tanimoto"]

            valid_count += 1
            results.append({
                "row_index": idx + 1,
                "raw_smiles": raw_val,
                "is_valid": True,
                "canonical_smiles": can_smiles,
                "prediction_class": pred_class,
                "probability": round(prob, 4),
                "applicability_domain": ad_status,
                "max_tanimoto": max_tanimoto,
                "molecular_weight": desc["molecular_weight"],
                "logp": desc["logp"],
                "tpsa": desc["tpsa"],
                "hbd": desc["hbd"],
                "hba": desc["hba"]
            })
        except Exception as ex:
            invalid_count += 1
            results.append({
                "row_index": idx + 1,
                "raw_smiles": raw_val,
                "is_valid": False,
                "error_message": f"Processing error: {str(ex)}"
            })

    return {
        "total_rows": total_rows,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "smiles_column": smiles_col,
        "results": results,
        "disclaimer": "This tool provides computational predictions for research and educational purposes. It is not a clinical diagnostic tool and should not be used as the sole basis for drug-safety decisions."
    }


# Mount built frontend static files if available
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
if DIST_DIR.exists() and (DIST_DIR / "index.html").exists():
    if (DIST_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Do not intercept /api routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        file_path = DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")


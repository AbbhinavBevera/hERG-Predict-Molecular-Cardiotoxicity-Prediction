"""
Explainability module for hERG risk prediction.
Extracts active ECFP4 fingerprint bits, computes their directional contribution to the predicted probability,
maps bits back to 2D atom indices for structure highlighting, and generates scientific interpretations.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from rdkit import Chem
from backend.models.cheminformatics import (
    compute_ecfp4_fingerprint,
    get_substructure_atoms_for_bit
)


# Mechanistic rules associated with classic hERG pharmacophore features
PHARMACOPHORE_RULES = [
    {
        "name": "Basic Tertiary/Secondary Amine",
        "pattern": "[#7;!$(N*=O);!$(N~S);!$(N~C=O)]",
        "mechanistic_note": "Positively charged or protonatable nitrogen at physiological pH. Frequently forms key cation-pi or electrostatic interactions with aromatic residues (Tyr652, Phe656) in the central cavity pore.",
        "risk_trend": "increases_risk"
    },
    {
        "name": "Hydrophobic / Aromatic Motif",
        "pattern": "a1aaaaa1",
        "mechanistic_note": "Aromatic rings engage in strong hydrophobic and pi-stacking interactions with Phe656 residues lining the deep cavity of the hERG potassium channel.",
        "risk_trend": "increases_risk"
    },
    {
        "name": "Lipophilic Halogenation (Cl / Br / CF3)",
        "pattern": "c[Cl,Br,I]",
        "mechanistic_note": "Increases overall molecular lipophilicity (LogP) and fits tightly into hydrophobic pockets of the S6 transmembrane domain.",
        "risk_trend": "increases_risk"
    },
    {
        "name": "Polar / Anionic Functionality (Carboxylate / Sulfonate)",
        "pattern": "[C,S](=[OX1])[O-]",
        "mechanistic_note": "Strong negative charges or elevated topological polar surface area (TPSA > 90) strongly disfavor entry into the hydrophobic pore cavity, attenuating hERG blockade.",
        "risk_trend": "decreases_risk"
    },
    {
        "name": "High Hydrophilic Density / Polyol",
        "pattern": "[OX2H][CX4][CX4][OX2H]",
        "mechanistic_note": "Hydrophilic and hydrogen-bonding motifs raise desolvation penalties, reducing channel pore affinity.",
        "risk_trend": "decreases_risk"
    }
]


def explain_prediction(
    mol: Chem.Mol,
    rf_model: Any,
    base_prob: float,
    top_n: int = 6
) -> List[Dict[str, Any]]:
    """
    Computes feature attribution for active bits of the molecule via counterfactual perturbation.
    For each on-bit i, turns it off to measure delta = P(original) - P(without_bit_i).
    """
    fp_vec, bit_info = compute_ecfp4_fingerprint(mol, return_bit_info=True)
    on_bits = np.where(fp_vec == 1)[0]

    if len(on_bits) == 0:
        return []

    # Fast vectorized perturbed batch:
    # Build a matrix of shape (len(on_bits), 1024) where each row is fp_vec with bit k flipped to 0
    perturbed_fps = np.repeat(fp_vec[np.newaxis, :], len(on_bits), axis=0)
    for idx, bit_id in enumerate(on_bits):
        perturbed_fps[idx, bit_id] = 0

    try:
        # Predict probabilities of blocker (class 1)
        perturbed_probs = rf_model.predict_proba(perturbed_fps)[:, 1]
        deltas = base_prob - perturbed_probs
    except Exception:
        # Fallback if model doesn't support batch predict_proba
        deltas = np.zeros(len(on_bits))

    # Rank by absolute impact magnitude
    scored_bits = []
    for idx, bit_id in enumerate(on_bits):
        delta = float(deltas[idx])
        atoms = get_substructure_atoms_for_bit(mol, int(bit_id), bit_info) if bit_info else []

        # Extract fragment SMILES if atoms available
        sub_smiles = ""
        if atoms:
            try:
                submol = Chem.PathToSubmol(mol, Chem.FindAtomEnvironmentOfRadiusN(
                    mol,
                    bit_info[bit_id][0][1],
                    bit_info[bit_id][0][0]
                )) if bit_info and bit_id in bit_info else None
                if submol:
                    sub_smiles = Chem.MolToSmiles(submol)
            except Exception:
                sub_smiles = ""

        scored_bits.append({
            "bit_id": int(bit_id),
            "delta": round(delta, 4),
            "abs_delta": abs(delta),
            "atom_indices": atoms,
            "sub_smiles": sub_smiles,
            "radius": bit_info[bit_id][0][1] if bit_info and bit_id in bit_info else 0
        })

    # Sort by absolute impact magnitude descending
    scored_bits.sort(key=lambda x: x["abs_delta"], reverse=True)
    top_bits = scored_bits[:top_n]

    # Annotate with mechanistic pharmacophore commentary
    for item in top_bits:
        direction = "increases_risk" if item["delta"] > 0 else "decreases_risk"
        item["direction"] = direction
        item["contribution_percentage"] = round(item["delta"] * 100, 1)

        # Match structural patterns
        matched_annotation = None
        for rule in PHARMACOPHORE_RULES:
            pat = Chem.MolFromSmarts(rule["pattern"])
            if pat and mol.HasSubstructMatch(pat):
                # Check overlap with item atom_indices
                matches = mol.GetSubstructMatches(pat)
                for m in matches:
                    if set(m).intersection(set(item["atom_indices"])):
                        matched_annotation = rule
                        break
            if matched_annotation:
                break

        if matched_annotation:
            item["fragment_type"] = matched_annotation["name"]
            item["mechanistic_explanation"] = matched_annotation["mechanistic_note"]
        else:
            if direction == "increases_risk":
                item["fragment_type"] = f"Lipophilic / Cavity-interfacing bit #{item['bit_id']}"
                item["mechanistic_explanation"] = "Local atomic neighborhood contributes positively toward hERG channel blockade liability."
            else:
                item["fragment_type"] = f"Polar / Attenuating bit #{item['bit_id']}"
                item["mechanistic_explanation"] = "Local atomic environment lowers predicted binding propensity within the hERG central cavity."

    return top_bits

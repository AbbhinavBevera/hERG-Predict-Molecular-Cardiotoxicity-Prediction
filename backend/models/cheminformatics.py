"""
Cheminformatics utility module using RDKit for:
- SMILES validation and canonicalization
- Salt removal & standardization
- Molecular property calculation (MW, LogP, TPSA, HBD, HBA, RotBonds)
- ECFP4 (Morgan radius 2, 1024-bit) fingerprint generation with bitInfo
- 2D SVG molecular structure rendering with substructure atom/bond highlights
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D


# Preprocessing configuration constants (must match model_metadata.json)
FP_RADIUS = 2  # ECFP4
FP_N_BITS = 1024
USE_CHIRALITY = False


def validate_and_parse_smiles(smiles: str) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """
    Validates a SMILES string. Returns (mol, None) if valid, or (None, error_message) if invalid.
    """
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return None, "Empty or non-string SMILES supplied."

    clean_smiles = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return None, f"RDKit could not parse the SMILES: '{clean_smiles}'. Check chemical valence and syntax."
        return mol, None
    except Exception as e:
        return None, f"Parsing exception: {str(e)}"


def standardize_molecule(mol: Chem.Mol) -> Tuple[Chem.Mol, str, str]:
    """
    Standardizes a molecule:
    - Retains the largest organic fragment (desalting)
    - Sanitizes molecule
    - Returns (standardized_mol, canonical_smiles, inchi_key)
    """
    # Desalt: keep fragment with largest number of heavy atoms
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if frags:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms() if m else 0)

    # Sanitize
    Chem.SanitizeMol(mol)
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    inchi_key = Chem.MolToInchiKey(mol)
    return mol, canonical_smiles, inchi_key


def calculate_descriptors(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Computes key physicochemical descriptors for drug-likeness & hERG risk profiling.
    """
    mw = round(float(Descriptors.MolWt(mol)), 2)
    logp = round(float(Descriptors.MolLogP(mol)), 2)
    tpsa = round(float(Descriptors.TPSA(mol)), 2)
    hbd = int(Descriptors.NumHDonors(mol))
    hba = int(Descriptors.NumHAcceptors(mol))
    rot_bonds = int(Descriptors.NumRotatableBonds(mol))
    aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    formal_charge = int(Chem.GetFormalCharge(mol))
    formula = rdMolDescriptors.CalcMolFormula(mol)

    # Lipinski Rule of 5 check
    lipinski_violations = sum([
        mw > 500,
        logp > 5.0,
        hbd > 5,
        hba > 10
    ])

    return {
        "formula": formula,
        "molecular_weight": mw,
        "logp": logp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rot_bonds,
        "aromatic_rings": aromatic_rings,
        "heavy_atoms": heavy_atoms,
        "formal_charge": formal_charge,
        "lipinski_violations": lipinski_violations,
    }


def compute_ecfp4_fingerprint(
    mol: Chem.Mol,
    radius: int = FP_RADIUS,
    n_bits: int = FP_N_BITS,
    use_chirality: bool = USE_CHIRALITY,
    return_bit_info: bool = False
) -> Tuple[np.ndarray, Optional[Dict[int, List[Tuple[int, int]]]]]:
    """
    Computes Morgan ECFP4 fingerprint as a binary numpy array of length n_bits.
    Optionally returns bitInfo dictionary mapping bit_id -> [(atom_idx, radius), ...].
    """
    if return_bit_info:
        bit_info: Dict[int, List[Tuple[int, int]]] = {}
        # For bitInfo tracking, AllChem.GetMorganFingerprintAsBitVect provides exact atomic mapping
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits, useChirality=use_chirality, bitInfo=bit_info
        )
    else:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
        bit_info = None

    arr = np.zeros((n_bits,), dtype=np.int8)
    for bit_id in fp.GetOnBits():
        arr[bit_id] = 1

    return arr, (bit_info if return_bit_info else None)


def get_substructure_atoms_for_bit(
    mol: Chem.Mol,
    bit_id: int,
    bit_info: Dict[int, List[Tuple[int, int]]]
) -> List[int]:
    """
    Given a bit_id and bit_info from Morgan fingerprint, finds all atom indices
    belonging to the substructure environment centered at the bit's root atom.
    """
    if bit_id not in bit_info or not bit_info[bit_id]:
        return []

    atoms_involved = set()
    for root_atom_idx, radius in bit_info[bit_id]:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, root_atom_idx)
        amap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        for orig_idx in amap.keys():
            atoms_involved.add(orig_idx)
        atoms_involved.add(root_atom_idx)

    return sorted(list(atoms_involved))


def render_molecule_svg(
    mol: Chem.Mol,
    highlight_atoms: Optional[List[int]] = None,
    highlight_bonds: Optional[List[int]] = None,
    highlight_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
    width: int = 400,
    height: int = 300,
    dark_mode: bool = False,
) -> str:
    """
    Renders high-quality 2D SVG representation with optional substructure atom/bond highlights.
    """
    # Clone mol to avoid altering original coordinates
    mol_copy = Chem.Mol(mol)
    AllChem.Compute2DCoords(mol_copy)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    draw_options = drawer.drawOptions()
    draw_options.clearBackground = False
    draw_options.addStereoAnnotation = True
    draw_options.bondLineWidth = 2.4

    highlight_atoms = highlight_atoms or []
    highlight_bonds = highlight_bonds or []

    # If bonds aren't explicitly passed but atoms are, highlight bonds connecting highlighted atoms
    if highlight_atoms and not highlight_bonds:
        atom_set = set(highlight_atoms)
        highlight_bonds = []
        for bond in mol_copy.GetBonds():
            if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
                highlight_bonds.append(bond.GetIdx())

    if highlight_colors:
        drawer.DrawMolecule(
            mol_copy,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
            highlightBonds=highlight_bonds
        )
    elif highlight_atoms:
        drawer.DrawMolecule(
            mol_copy,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds
        )
    else:
        drawer.DrawMolecule(mol_copy)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Clean up SVG XML header and background rects so it embeds cleanly in HTML
    svg = re.sub(r'<\?xml[^>]*\?>', '', svg)
    svg = re.sub(r'<rect[^>]*>\s*</rect>', '', svg)

    if dark_mode:
        # Convert black bonds and text (both CSS styles and XML attributes) to crisp high-contrast silver-white
        svg = re.sub(r'stroke:\s*#000000', 'stroke:#f0f6fc', svg)
        svg = re.sub(r'fill:\s*#000000', 'fill:#f0f6fc', svg)
        svg = re.sub(r"stroke=['\"]#?000000['\"]", "stroke='#f0f6fc'", svg)
        svg = re.sub(r"fill=['\"]#?000000['\"]", "fill='#f0f6fc'", svg)
        svg = re.sub(r'stroke:\s*black', 'stroke:#f0f6fc', svg)
        svg = re.sub(r'fill:\s*black', 'fill:#f0f6fc', svg)
        # Brighten dark blue Nitrogen to vibrant electric cyan/sky blue
        svg = re.sub(r'#0000FF', '#38bdf8', svg, flags=re.IGNORECASE)
        svg = re.sub(r'#0000CD', '#38bdf8', svg, flags=re.IGNORECASE)

    return svg.strip()

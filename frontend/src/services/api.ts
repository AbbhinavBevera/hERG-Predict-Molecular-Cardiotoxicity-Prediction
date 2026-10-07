import {
  PredictionResponse,
  ReferenceMolecule,
  BenchmarkMetricsResponse,
  BatchResponse,
} from "../types";

const API_BASE = "/api";

export async function predictMolecule(
  smiles: string,
  modelChoice: "random_forest" | "gnn" | "consensus" = "consensus",
  darkMode: boolean = true
): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles, model_choice: modelChoice, dark_mode: darkMode }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    const message =
      errorData.detail?.message ||
      errorData.detail?.error ||
      errorData.detail ||
      `Server error (${res.status})`;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }

  return res.json();
}

export async function renderCustomSvg(
  smiles: string,
  highlightAtoms: number[],
  darkMode: boolean = true
): Promise<string> {
  const res = await fetch(`${API_BASE}/render-svg`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      smiles,
      highlight_atoms: highlightAtoms,
      dark_mode: darkMode,
    }),
  });

  if (!res.ok) {
    throw new Error(`Failed to render SVG: ${res.statusText}`);
  }

  const data = await res.json();
  return data.svg;
}

const FALLBACK_EXAMPLES: ReferenceMolecule[] = [
  {
    name: "Terfenadine",
    category: "Antihistamine (Known Potent Blocker)",
    clinical_context: "Withdrawn worldwide due to lethal QT prolongation and Torsades de Pointes.",
    known_risk: "High Risk (Potent Blocker)",
    smiles: "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c4ccccc4)CC2)cc1",
  },
  {
    name: "Astemizole",
    category: "Antihistamine (Known Potent Blocker)",
    clinical_context: "Second-generation H1 antagonist withdrawn due to fatal arrhythmias.",
    known_risk: "High Risk (Potent Blocker)",
    smiles: "COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc5ccc(F)cc5)CC2)cc1",
  },
  {
    name: "Cisapride",
    category: "Gastroprokinetic (Known Blocker)",
    clinical_context: "Restricted/withdrawn after hundreds of reported cardiac deaths.",
    known_risk: "High Risk (Blocker)",
    smiles: "COc1cc(Cl)c(N)cc1C(=O)NC2CCN(CCCOC3ccc(F)cc3)C(OC)C2",
  },
  {
    name: "Haloperidol",
    category: "Antipsychotic (Known Blocker)",
    clinical_context: "High-affinity D2 antagonist with dose-dependent QT prolongation liability.",
    known_risk: "High Risk (Blocker)",
    smiles: "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c3ccc(Cl)cc3",
  },
  {
    name: "Dofetilide",
    category: "Class III Antiarrhythmic (Reference Blocker)",
    clinical_context: "Selective IKr blocker requiring 3-day continuous in-hospital ECG telemetry.",
    known_risk: "High Risk (Strong Blocker)",
    smiles: "CN(CCc1ccc(NS(=O)(=O)C)cc1)CCc2ccc(NS(=O)(=O)C)cc2",
  },
  {
    name: "Aspirin",
    category: "Analgesic / NSAID (Non-Blocker)",
    clinical_context: "Nonsteroidal anti-inflammatory drug. Devoid of basic amines, negative for hERG blockade.",
    known_risk: "Low Risk (Non-Blocker)",
    smiles: "CC(=O)Oc1ccccc1C(=O)O",
  },
  {
    name: "Caffeine",
    category: "CNS Stimulant (Non-Blocker)",
    clinical_context: "Methylxanthine adenosine receptor antagonist. Devoid of hERG liability.",
    known_risk: "Low Risk (Non-Blocker)",
    smiles: "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
  },
  {
    name: "Amoxicillin",
    category: "Beta-Lactam Antibiotic (Non-Blocker)",
    clinical_context: "Broad-spectrum hydrophilic aminopenicillin, zero hERG affinity, safe cardiac profile.",
    known_risk: "Low Risk (Non-Blocker)",
    smiles: "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
  },
  {
    name: "Metformin",
    category: "Antidiabetic (Non-Blocker)",
    clinical_context: "Biguanide first-line type 2 diabetes agent with no hERG channel inhibition.",
    known_risk: "Low Risk (Non-Blocker)",
    smiles: "CN(C)C(=N)NC(=N)N",
  },
  {
    name: "Ibuprofen",
    category: "Analgesic / NSAID (Non-Blocker)",
    clinical_context: "Propionic acid NSAID. Lacks basic amine pharmacophore, negative for hERG blockade.",
    known_risk: "Low Risk (Non-Blocker)",
    smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
  },
];

export async function getReferenceExamples(): Promise<ReferenceMolecule[]> {
  try {
    const res = await fetch(`${API_BASE}/examples`);
    if (res.ok) return await res.json();
  } catch (_) {}
  return FALLBACK_EXAMPLES;
}

export async function getBenchmarkMetrics(): Promise<BenchmarkMetricsResponse> {
  const base = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (res.ok) return await res.json();
  } catch (_) {}

  const res = await fetch(`${base}/benchmark_metrics.json`);
  if (!res.ok) {
    throw new Error("Failed to load benchmark metrics");
  }
  return res.json();
}

export async function submitBatchPrediction(
  file: File,
  modelChoice: string = "consensus"
): Promise<BatchResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/batch?model_choice=${modelChoice}`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Batch prediction request failed");
  }

  return res.json();
}

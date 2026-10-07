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

export async function getReferenceExamples(): Promise<ReferenceMolecule[]> {
  const res = await fetch(`${API_BASE}/examples`);
  if (!res.ok) {
    throw new Error("Failed to load reference examples");
  }
  return res.json();
}

export async function getBenchmarkMetrics(): Promise<BenchmarkMetricsResponse> {
  const res = await fetch(`${API_BASE}/metrics`);
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

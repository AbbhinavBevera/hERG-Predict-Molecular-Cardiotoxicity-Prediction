export interface FragmentAttribution {
  bit_id: number;
  delta: number;
  abs_delta: number;
  atom_indices: number[];
  sub_smiles: string;
  radius: number;
  direction: "increases_risk" | "decreases_risk";
  contribution_percentage: number;
  fragment_type: string;
  mechanistic_explanation: string;
}

export interface ApplicabilityDomainResult {
  status: "In-Domain" | "Borderline" | "Out-of-Domain" | "Unknown";
  max_tanimoto: number;
  knn_tanimoto: number;
  is_reliable: boolean;
  warning?: string | null;
  explanation?: string;
  thresholds?: {
    in_domain: number;
    borderline: number;
  };
}

export interface MolecularDescriptors {
  formula: string;
  molecular_weight: number;
  logp: number;
  tpsa: number;
  hbd: number;
  hba: number;
  rotatable_bonds: number;
  aromatic_rings: number;
  heavy_atoms: number;
  formal_charge: number;
  lipinski_violations: number;
}

export interface PredictionDetail {
  prediction_class: string;
  is_blocker: boolean;
  probability: number;
  percentage: number;
  confidence_percentage: number;
  risk_tier: string;
  model_used: string;
  model_breakdown: {
    random_forest_probability: number;
    gnn_probability: number;
  };
}

export interface PredictionResponse {
  smiles: string;
  canonical_smiles: string;
  inchi_key: string;
  prediction: PredictionDetail;
  applicability_domain: ApplicabilityDomainResult;
  explainability: {
    top_fragments: FragmentAttribution[];
    pharmacophore_summary: string;
  };
  descriptors: MolecularDescriptors;
  svg: string;
  disclaimer: string;
}

export interface ReferenceMolecule {
  name: string;
  category: string;
  clinical_context: string;
  known_risk: string;
  smiles: string;
}

export interface BatchItemResult {
  row_index: number;
  raw_smiles: string;
  is_valid: boolean;
  error_message?: string;
  canonical_smiles?: string;
  prediction_class?: string;
  probability?: number;
  applicability_domain?: string;
  max_tanimoto?: number;
  molecular_weight?: number;
  logp?: number;
  tpsa?: number;
  hbd?: number;
  hba?: number;
}

export interface BatchResponse {
  total_rows: number;
  valid_count: number;
  invalid_count: number;
  smiles_column: string;
  results: BatchItemResult[];
  disclaimer: string;
}

export interface CurvePoint {
  fpr?: number;
  tpr?: number;
  recall?: number;
  precision?: number;
}

export interface EvaluationSplitMetrics {
  roc_auc: number;
  pr_auc: number;
  accuracy: number;
  balanced_accuracy: number;
  precision: number;
  recall_sensitivity: number;
  specificity: number;
  f1: number;
  mcc: number;
  confusion_matrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
  roc_curve: Array<{ fpr: number; tpr: number }>;
  pr_curve: Array<{ recall: number; precision: number }>;
}

export interface ModelMetricsDetail {
  name: string;
  description: string;
  internal_test: EvaluationSplitMetrics;
  external_validation: EvaluationSplitMetrics;
}

export interface BenchmarkMetricsResponse {
  dataset_metadata: {
    internal_test_size: number;
    internal_test_source: string;
    external_test_size: number;
    external_test_source: string;
    evaluation_timestamp: string;
  };
  models: {
    random_forest: ModelMetricsDetail;
    molecular_gnn?: ModelMetricsDetail;
  };
}

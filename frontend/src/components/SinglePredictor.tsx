import React, { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";
import type { PredictionResponse, ReferenceMolecule, FragmentAttribution } from "../types";
import { predictMolecule, renderCustomSvg } from "../services/api";
import { predictInBrowser } from "../services/browserInference";
import { MolecularViewer } from "./MolecularViewer";
import { ApplicabilityDomainBadge } from "./ApplicabilityDomainBadge";
import { FragmentExplainer } from "./FragmentExplainer";

interface Props {
  examples: ReferenceMolecule[];
}

export const SinglePredictor: React.FC<Props> = ({ examples }) => {
  const [smiles, setSmiles] = useState<string>(
    "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c4ccccc4)CC2)cc1" // Terfenadine
  );
  const [selectedModel, setSelectedModel] = useState<"consensus" | "random_forest" | "gnn" | "browser_rdkit">("consensus");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  // Highlighting state
  const [activeSvg, setActiveSvg] = useState<string>("");
  const [highlightedAtoms, setHighlightedAtoms] = useState<number[]>([]);
  const [activeFragment, setActiveFragment] = useState<FragmentAttribution | null>(null);

  // Render molecule and prediction immediately on initial mount
  useEffect(() => {
    handlePredict("CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c4ccccc4)CC2)cc1");
  }, []);

  const handlePredict = async (inputSmiles?: string) => {
    const targetSmiles = inputSmiles !== undefined ? inputSmiles : smiles;
    if (!targetSmiles.trim()) {
      setError("Please enter a valid SMILES string.");
      return;
    }

    setLoading(true);
    setError(null);
    setActiveFragment(null);
    setHighlightedAtoms([]);

    try {
      let data: PredictionResponse;
      if (selectedModel === "browser_rdkit") {
        data = await predictInBrowser(targetSmiles.trim());
      } else {
        try {
          data = await predictMolecule(targetSmiles.trim(), selectedModel);
        } catch (serverErr) {
          // If server is unreachable (e.g. GitHub Pages static deployment), evaluate client-side with RDKit.js
          console.warn("Server unavailable, using in-browser RDKit.js:", serverErr);
          data = await predictInBrowser(targetSmiles.trim());
        }
      }
      setResult(data);
      setActiveSvg(data.svg);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || "Prediction failed.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectExample = (ex: ReferenceMolecule) => {
    setSmiles(ex.smiles);
    setError(null);
    handlePredict(ex.smiles);
  };

  const handleHighlightFragment = async (frag: FragmentAttribution) => {
    if (!result) return;
    setActiveFragment(frag);
    setHighlightedAtoms(frag.atom_indices);

    try {
      const newSvg = await renderCustomSvg(result.smiles, frag.atom_indices, true);
      setActiveSvg(newSvg);
    } catch (e) {
      console.error(e);
    }
  };

  const handleResetHighlight = async () => {
    if (!result) return;
    setActiveFragment(null);
    setHighlightedAtoms([]);
    try {
      const resetSvg = await renderCustomSvg(result.smiles, [], true);
      setActiveSvg(resetSvg);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="space-y-6">
      {/* Input Box - Liquid Glass Card */}
      <div className="liquid-glass-card rounded-xl p-4 sm:p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <label className="text-xs font-semibold text-[#e6edf3]">
            SMILES Input:
          </label>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-[#8b949e]">Model:</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as any)}
              className="border border-white/10 rounded-lg px-2.5 py-1 text-xs bg-[#0d1117]/80 text-[#e6edf3] focus:outline-none focus:border-[#58a6ff] transition-colors"
            >
              <option value="consensus">Consensus (Server RF + GNN)</option>
              <option value="random_forest">Random Forest (Server ECFP4)</option>
              <option value="gnn">Graph Neural Network (Server PyTorch)</option>
              <option value="browser_rdkit">Client-Side (RDKit.js WASM + JSON Weights)</option>
            </select>
          </div>
        </div>

        <div className="space-y-2">
          <textarea
            rows={2}
            value={smiles}
            onChange={(e) => {
              setSmiles(e.target.value);
              setError(null);
            }}
            placeholder="Enter SMILES, e.g. CC(=O)Oc1ccccc1C(=O)O"
            className="w-full border border-white/10 rounded-lg p-2.5 text-xs font-mono bg-[#0d1117]/80 text-[#e6edf3] focus:outline-none focus:border-[#58a6ff] focus:ring-1 focus:ring-[#58a6ff]/40 transition-colors"
          />

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button
                onClick={() => handlePredict()}
                disabled={loading || !smiles.trim()}
                className="bg-[#238636] hover:bg-[#2ea043] active:scale-95 text-white px-4 py-2 rounded-lg text-xs font-semibold disabled:opacity-50 transition-all duration-200 flex items-center gap-1.5 shadow-[0_0_15px_rgba(35,134,54,0.35)] hover:shadow-[0_0_22px_rgba(46,160,67,0.55)] cursor-pointer"
              >
                {loading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                <span>{loading ? "Calculating..." : "Predict"}</span>
              </button>
              {smiles && (
                <button
                  onClick={() => setSmiles("")}
                  className="border border-white/10 px-3 py-2 rounded-lg text-xs text-[#8b949e] hover:bg-[#21262d]/60 hover:text-[#e6edf3] transition-colors cursor-pointer"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="p-3 bg-[#25171c]/90 border border-[#f85149]/40 text-[#ff7b72] text-xs rounded-lg animate-fade-in">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Example Pill Buttons */}
        {examples.length > 0 && (
          <div className="pt-3 border-t border-white/10 space-y-2">
            <span className="text-xs font-semibold text-[#8b949e] block">
              Reference Drugs (Click to test):
            </span>
            <div className="flex flex-wrap gap-2">
              {examples.map((ex) => {
                const isBlocker = ex.known_risk.includes("Blocker") && !ex.known_risk.includes("Non");
                const isSelected = smiles === ex.smiles;
                return (
                  <button
                    key={ex.name}
                    onClick={() => handleSelectExample(ex)}
                    className={`inline-flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs font-medium cursor-pointer liquid-glass-pill ${
                      isSelected
                        ? "selected border-[#58a6ff] text-[#e6edf3]"
                        : "text-[#8b949e] hover:text-[#e6edf3]"
                    }`}
                  >
                    <span
                      className={`w-2 h-2 rounded-full shrink-0 shadow-xs ${
                        isBlocker ? "bg-[#f85149] shadow-[0_0_6px_rgba(248,81,73,0.6)]" : "bg-[#3fb950] shadow-[0_0_6px_rgba(63,185,80,0.6)]"
                      }`}
                    />
                    <span>{ex.name}</span>
                    <span className="text-[10px] text-[#8b949e]">
                      {isBlocker ? "(Blocker)" : "(Safe)"}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-4 animate-fade-in">
          {/* Summary Box with Liquid Flow Gauge */}
          <div className="liquid-glass-card rounded-xl p-4 sm:p-5 space-y-3.5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <div className="text-xs text-[#8b949e]">Prediction Result</div>
                <div className={`text-lg font-bold flex items-center gap-2 ${result.prediction.is_blocker ? "text-[#f85149]" : "text-[#3fb950]"}`}>
                  <span>{result.prediction.prediction_class}</span>
                  <span
                    className={`inline-block w-2.5 h-2.5 rounded-full ${
                      result.prediction.is_blocker
                        ? "bg-[#f85149] shadow-[0_0_8px_rgba(248,81,73,0.8)]"
                        : "bg-[#3fb950] shadow-[0_0_8px_rgba(63,185,80,0.8)]"
                    }`}
                  />
                </div>
              </div>

              <div className="flex items-center gap-5 text-xs font-mono">
                <div>
                  <span className="text-[#8b949e] block">Probability:</span>
                  <strong className="text-base font-bold text-[#e6edf3]">
                    {result.prediction.percentage.toFixed(1)}%
                  </strong>
                </div>
                <div>
                  <span className="text-[#8b949e] block">Random Forest:</span>
                  <span className="text-[#e6edf3]">{(result.prediction.model_breakdown.random_forest_probability * 100).toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-[#8b949e] block">GNN:</span>
                  <span className="text-[#e6edf3]">{(result.prediction.model_breakdown.gnn_probability * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Smooth liquid flow probability gauge */}
            <div className="space-y-1">
              <div className="w-full bg-[#0d1117]/80 h-2.5 rounded-full overflow-hidden border border-white/10 p-0.5">
                <div
                  className={`h-full rounded-full transition-all duration-700 ease-out shadow-xs ${
                    result.prediction.is_blocker
                      ? "liquid-flow-red"
                      : "liquid-flow-green"
                  }`}
                  style={{ width: `${result.prediction.percentage}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-[#8b949e] font-mono">
                <span>0% Safe</span>
                <span>Cutoff: 50%</span>
                <span>100% Blocker</span>
              </div>
            </div>

            <div className="text-[11px] text-[#8b949e] pt-1">
              Evaluated with {result.prediction.model_used}. Cutoff threshold is 50.0%.
            </div>
          </div>

          {/* Grid: 2D Viewer + Descriptors & AD */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <MolecularViewer
              svg={activeSvg}
              descriptors={result.descriptors}
              highlightedCount={highlightedAtoms.length}
              onResetHighlight={handleResetHighlight}
              activeFragmentName={activeFragment?.fragment_type}
            />

            <div className="space-y-4">
              <ApplicabilityDomainBadge ad={result.applicability_domain} />

              <div className="liquid-glass rounded-xl p-4 text-xs">
                <div className="font-semibold text-[#e6edf3] mb-2">Physicochemical Descriptors</div>
                <table>
                  <tbody>
                    <tr>
                      <td className="text-[#8b949e]">Molecular Weight</td>
                      <td className="font-mono text-[#e6edf3]">{result.descriptors.molecular_weight} g/mol</td>
                      <td className="text-[#8b949e]">LogP</td>
                      <td className="font-mono text-[#e6edf3]">{result.descriptors.logp}</td>
                    </tr>
                    <tr>
                      <td className="text-[#8b949e]">TPSA</td>
                      <td className="font-mono text-[#e6edf3]">{result.descriptors.tpsa} &#197;&sup2;</td>
                      <td className="text-[#8b949e]">HBD / HBA</td>
                      <td className="font-mono text-[#e6edf3]">{result.descriptors.hbd} / {result.descriptors.hba}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Fragment Explainer Table */}
          <FragmentExplainer
            fragments={result.explainability.top_fragments}
            activeBitId={activeFragment?.bit_id}
            onSelectFragment={handleHighlightFragment}
            pharmacophoreSummary={result.explainability.pharmacophore_summary}
          />
        </div>
      )}
    </div>
  );
};

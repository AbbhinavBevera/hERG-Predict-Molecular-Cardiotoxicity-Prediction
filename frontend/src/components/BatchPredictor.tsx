import React, { useState } from "react";
import type { BatchResponse } from "../types";
import { submitBatchPrediction } from "../services/api";

export const BatchPredictor: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [modelChoice, setModelChoice] = useState<string>("consensus");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [batchData, setBatchData] = useState<BatchResponse | null>(null);

  const handleDownloadSample = () => {
    const sampleCsv = `smiles,name
CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c4ccccc4)CC2)cc1,Terfenadine
COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc5ccc(F)cc5)CC2)cc1,Astemizole
COc1cc(Cl)c(N)cc1C(=O)NC2CCN(CCCOC3ccc(F)cc3)C(OC)C2,Cisapride
OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c3ccc(Cl)cc3,Haloperidol
CN(CCc1ccc(NS(=O)(=O)C)cc1)CCc2ccc(NS(=O)(=O)C)cc2,Dofetilide
CC(=O)Oc1ccccc1C(=O)O,Aspirin
Cn1cnc2c1c(=O)n(C)c(=O)n2C,Caffeine
CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O,Amoxicillin
CN(C)C(=N)NC(=N)N,Metformin
INVALID_SMILES,BadMolecule
`;
    const blob = new Blob([sampleCsv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sample_smiles.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Please select a CSV file.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await submitBatchPrediction(file, modelChoice);
      setBatchData(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || "Batch prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleExportResults = () => {
    if (!batchData) return;

    const headers = [
      "Row",
      "SMILES",
      "Valid",
      "Result",
      "Probability",
      "Domain",
      "MW",
      "LogP",
      "Error"
    ];

    const rows = batchData.results.map((r) => [
      r.row_index,
      `"${r.raw_smiles.replace(/"/g, '""')}"`,
      r.is_valid ? "Yes" : "No",
      r.prediction_class || "",
      r.probability !== undefined ? `${(r.probability * 100).toFixed(1)}%` : "",
      r.applicability_domain || "",
      r.molecular_weight ?? "",
      r.logp ?? "",
      r.error_message ? `"${r.error_message.replace(/"/g, '""')}"` : ""
    ]);

    const csvContent = [headers.join(","), ...rows.map((row) => row.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `herg_predictions_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="liquid-glass-card rounded-xl p-4 sm:p-5 space-y-4">
        <div>
          <h2 className="text-sm font-semibold text-[#e6edf3]">Batch SMILES Screening</h2>
          <p className="text-xs text-[#8b949e]">
            Upload a CSV file containing a column with SMILES codes (e.g. named <code>smiles</code>, <code>drug</code>, or <code>molecule</code>).
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs">
          <input
            type="file"
            accept=".csv"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                setFile(e.target.files[0]);
                setError(null);
              }
            }}
            className="border border-white/10 rounded-lg p-1.5 text-xs bg-[#0d1117]/80 text-[#e6edf3]"
          />

          <div className="flex items-center gap-1.5">
            <span className="text-[#8b949e]">Model:</span>
            <select
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
              className="border border-white/10 rounded-lg px-2.5 py-1 text-xs bg-[#0d1117]/80 text-[#e6edf3]"
            >
              <option value="consensus">Consensus (RF + GNN)</option>
              <option value="random_forest">Random Forest</option>
              <option value="gnn">Graph Neural Network</option>
            </select>
          </div>

          <button
            onClick={handleSubmit}
            disabled={!file || loading}
            className="bg-[#238636] hover:bg-[#2ea043] active:scale-95 text-white px-4 py-2 rounded-lg text-xs font-semibold disabled:opacity-50 transition-all duration-200 shadow-[0_0_15px_rgba(35,134,54,0.3)] hover:shadow-[0_0_20px_rgba(46,160,67,0.5)] cursor-pointer"
          >
            {loading ? "Screening..." : "Run Batch"}
          </button>

          <button
            onClick={handleDownloadSample}
            className="text-[#58a6ff] hover:underline text-xs cursor-pointer ml-auto"
          >
            Download sample CSV
          </button>
        </div>

        {error && (
          <div className="p-2.5 bg-[#25171c]/90 border border-[#f85149]/40 text-[#ff7b72] text-xs rounded-lg animate-fade-in">
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>

      {batchData && (
        <div className="liquid-glass rounded-xl p-4 sm:p-5 space-y-3 animate-fade-in">
          <div className="flex items-center justify-between text-xs pb-2 border-b border-white/10">
            <div className="text-[#8b949e]">
              Total rows: <strong className="text-[#e6edf3]">{batchData.total_rows}</strong> | Valid: <strong className="text-[#e6edf3]">{batchData.valid_count}</strong> | Invalid: <strong className="text-[#e6edf3]">{batchData.invalid_count}</strong>
            </div>

            <button
              onClick={handleExportResults}
              className="border border-white/10 bg-[#21262d]/80 hover:bg-[#30363d] px-3 py-1.5 rounded-lg text-xs text-[#e6edf3] font-medium transition-colors cursor-pointer"
            >
              Export CSV
            </button>
          </div>

          <div className="overflow-x-auto text-xs">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Input SMILES</th>
                  <th>Valid</th>
                  <th>Prediction</th>
                  <th>Probability</th>
                  <th>Domain</th>
                  <th>MW</th>
                  <th>LogP</th>
                </tr>
              </thead>
              <tbody>
                {batchData.results.map((item) => (
                  <tr key={item.row_index}>
                    <td className="font-mono text-[#8b949e]">{item.row_index}</td>
                    <td className="font-mono max-w-xs truncate text-[#e6edf3]" title={item.raw_smiles}>
                      {item.raw_smiles}
                    </td>
                    <td>{item.is_valid ? "Yes" : <span className="text-[#ff7b72]">No</span>}</td>
                    <td>
                      {item.is_valid ? (
                        <span className={item.prediction_class?.includes("Blocker") ? "text-[#f85149] font-medium" : "text-[#3fb950] font-medium"}>
                          {item.prediction_class}
                        </span>
                      ) : (
                        <span className="text-[#ff7b72] text-[11px]">{item.error_message}</span>
                      )}
                    </td>
                    <td className="font-mono text-[#e6edf3]">
                      {item.probability !== undefined ? `${(item.probability * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className="text-[#8b949e]">{item.applicability_domain || "—"}</td>
                    <td className="font-mono text-[#8b949e]">{item.molecular_weight ?? "—"}</td>
                    <td className="font-mono text-[#8b949e]">{item.logp ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

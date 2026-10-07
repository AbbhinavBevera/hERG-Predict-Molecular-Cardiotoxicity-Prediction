import React, { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import type { BenchmarkMetricsResponse } from "../types";
import { getBenchmarkMetrics } from "../services/api";

export const ModelComparison: React.FC = () => {
  const [metrics, setMetrics] = useState<BenchmarkMetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [splitView, setSplitView] = useState<"internal_test" | "external_validation">("internal_test");

  useEffect(() => {
    getBenchmarkMetrics()
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  if (loading || !metrics) {
    return (
      <div className="liquid-glass rounded-xl p-6 text-xs text-[#8b949e]">
        Loading benchmark metrics...
      </div>
    );
  }

  const rfMetrics = metrics.models.random_forest[splitView];
  const gnnMetrics = metrics.models.molecular_gnn ? metrics.models.molecular_gnn[splitView] : null;

  const combinedRocData = [];
  const rfRoc = rfMetrics.roc_curve;
  const gnnRoc = gnnMetrics ? gnnMetrics.roc_curve : [];
  const maxPts = Math.max(rfRoc.length, gnnRoc.length);

  for (let i = 0; i < maxPts; i++) {
    const rfPt = rfRoc[Math.min(i, rfRoc.length - 1)];
    const gnnPt = gnnRoc[Math.min(i, gnnRoc.length - 1)];
    combinedRocData.push({
      step: i,
      rfFpr: rfPt?.fpr ?? 0,
      rfTpr: rfPt?.tpr ?? 0,
      gnnFpr: gnnPt?.fpr ?? 0,
      gnnTpr: gnnPt?.tpr ?? 0,
    });
  }

  const rows = [
    { label: "ROC-AUC", rf: rfMetrics.roc_auc, gnn: gnnMetrics?.roc_auc },
    { label: "PR-AUC (Average Precision)", rf: rfMetrics.pr_auc, gnn: gnnMetrics?.pr_auc },
    { label: "Accuracy", rf: rfMetrics.accuracy, gnn: gnnMetrics?.accuracy },
    { label: "Balanced Accuracy", rf: rfMetrics.balanced_accuracy, gnn: gnnMetrics?.balanced_accuracy },
    { label: "Sensitivity (Recall)", rf: rfMetrics.recall_sensitivity, gnn: gnnMetrics?.recall_sensitivity },
    { label: "Specificity", rf: rfMetrics.specificity, gnn: gnnMetrics?.specificity },
    { label: "F1 Score", rf: rfMetrics.f1, gnn: gnnMetrics?.f1 },
    { label: "Matthews Correlation (MCC)", rf: rfMetrics.mcc, gnn: gnnMetrics?.mcc },
  ];

  return (
    <div className="space-y-6">
      <div className="liquid-glass-card rounded-xl p-4 sm:p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-[#e6edf3]">
              Model Benchmark: Random Forest vs Graph Neural Network
            </h2>
            <p className="text-xs text-[#8b949e]">
              Evaluated on identical data splits. Measured empirical metrics.
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="text-[#8b949e]">Dataset Split:</span>
            <select
              value={splitView}
              onChange={(e) => setSplitView(e.target.value as any)}
              className="border border-white/10 rounded-lg px-2.5 py-1 text-xs bg-[#0d1117]/80 text-[#e6edf3] focus:outline-none focus:border-[#58a6ff]"
            >
              <option value="internal_test">Internal Test Split (n=1,309)</option>
              <option value="external_validation">External Validation Split (n=182)</option>
            </select>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto text-xs">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Random Forest (ECFP4)</th>
                <th>Graph Neural Network (GNN)</th>
                <th>Difference</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const diff = r.gnn !== undefined ? r.gnn - r.rf : 0;
                return (
                  <tr key={r.label}>
                    <td className="text-[#e6edf3]">{r.label}</td>
                    <td className="font-mono text-[#58a6ff]">{r.rf.toFixed(4)}</td>
                    <td className="font-mono text-[#d2a8ff]">{r.gnn !== undefined ? r.gnn.toFixed(4) : "—"}</td>
                    <td className="font-mono text-[#8b949e]">
                      {diff > 0 ? `+${diff.toFixed(4)} (GNN)` : `${diff.toFixed(4)} (RF)`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ROC Chart */}
      <div className="liquid-glass rounded-xl p-4 sm:p-5 space-y-2">
        <div className="text-xs font-semibold text-[#e6edf3]">
          ROC Curve ({splitView === "internal_test" ? "Internal Test" : "External Validation"})
        </div>
        <div className="h-60 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={combinedRocData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.08)" />
              <XAxis dataKey="rfFpr" type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: "#8b949e" }} />
              <YAxis dataKey="rfTpr" type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: "#8b949e" }} />
              <Tooltip
                formatter={(val: any) => Number(val).toFixed(3)}
                contentStyle={{ backgroundColor: "rgba(22, 27, 34, 0.9)", borderColor: "rgba(255, 255, 255, 0.1)", borderRadius: "8px", color: "#e6edf3", fontSize: "11px", backdropFilter: "blur(8px)" }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <Line
                name={`Random Forest (AUC: ${rfMetrics.roc_auc.toFixed(3)})`}
                dataKey="rfTpr"
                stroke="#58a6ff"
                strokeWidth={2}
                dot={false}
              />
              {gnnMetrics && (
                <Line
                  name={`Molecular GNN (AUC: ${gnnMetrics.roc_auc.toFixed(3)})`}
                  dataKey="gnnTpr"
                  stroke="#d2a8ff"
                  strokeWidth={2}
                  strokeDasharray="4 2"
                  dot={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Deduplication check */}
      <div className="p-3.5 liquid-glass rounded-xl text-xs text-[#8b949e] space-y-1">
        <strong className="text-[#e6edf3]">Data Leakage Audit:</strong>
        <div>
          Before external evaluation, all 13,090 training InChIKeys were cross-referenced against the candidate external library (Doddareddy et al.). 456 identical compounds were found and removed. The remaining 182 compounds were strictly evaluated with 0% data leakage.
        </div>
      </div>
    </div>
  );
};

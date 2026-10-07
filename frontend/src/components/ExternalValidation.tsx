import React, { useEffect, useState } from "react";
import {
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Database,
  ArrowRight,
  TrendingDown,
  Layers,
  Sparkles,
  Info,
} from "lucide-react";
import { BenchmarkMetricsResponse } from "../types";
import { getBenchmarkMetrics } from "../services/api";

export const ExternalValidation: React.FC = () => {
  const [metrics, setMetrics] = useState<BenchmarkMetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

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
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-12 text-center text-slate-400 text-xs">
        Loading external validation audits and metrics...
      </div>
    );
  }

  const rfInternal = metrics.models.random_forest.internal_test;
  const rfExternal = metrics.models.random_forest.external_validation;
  const gnnInternal = metrics.models.molecular_gnn?.internal_test;
  const gnnExternal = metrics.models.molecular_gnn?.external_validation;

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-xl bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">
              Independent External Validation & Leakage Audit
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Evaluated on an unobserved chemical dataset from an independent laboratory source (Doddareddy et al., 2010).
            </p>
          </div>
        </div>
      </div>

      {/* Leakage Audit Flow Diagram */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
        <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
          Rigorous InChIKey-Based Deduplication & Zero-Leakage Protocol
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800 text-center">
            <span className="text-[11px] text-slate-400 uppercase font-semibold block mb-1">
              Primary Corpus
            </span>
            <strong className="text-base font-bold text-slate-800 dark:text-white block font-mono">
              13,090 Unique Mol
            </strong>
            <span className="text-[11px] text-slate-500 mt-1 block">
              Karim et al. (ChEMBL)
            </span>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800 text-center">
            <span className="text-[11px] text-slate-400 uppercase font-semibold block mb-1">
              Raw External Set
            </span>
            <strong className="text-base font-bold text-slate-800 dark:text-white block font-mono">
              638 Unique Mol
            </strong>
            <span className="text-[11px] text-slate-500 mt-1 block">
              Doddareddy et al. (TDC)
            </span>
          </div>

          <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900/50 text-center">
            <span className="text-[11px] text-rose-500 uppercase font-semibold block mb-1">
              Pruned Overlaps
            </span>
            <strong className="text-base font-bold text-rose-600 dark:text-rose-400 block font-mono">
              -456 Collisions
            </strong>
            <span className="text-[11px] text-rose-600/80 mt-1 block">
              Removed to prevent leakage
            </span>
          </div>

          <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900/50 text-center">
            <span className="text-[11px] text-emerald-600 uppercase font-semibold block mb-1">
              Independent Test Set
            </span>
            <strong className="text-base font-bold text-emerald-600 dark:text-emerald-400 block font-mono">
              182 Clean Mol
            </strong>
            <span className="text-[11px] text-emerald-600/80 mt-1 block">
              Strictly zero train overlap
            </span>
          </div>
        </div>

        <div className="mt-4 p-3 rounded-xl bg-slate-50 dark:bg-slate-950/40 text-xs text-slate-600 dark:text-slate-400 leading-relaxed border border-slate-200 dark:border-slate-800">
          <strong className="text-slate-800 dark:text-slate-200">Scientific Integrity Guarantee:</strong>{" "}
          Canonical 27-character InChIKeys were calculated for every molecule in both databases. 456 molecules in the candidate external set that were already present in the training, validation, or test subsets of the primary dataset were quarantined and removed. The remaining 182 molecules (92 active blockers, 90 inactive non-blockers) represent an entirely unobserved chemical space.
        </div>
      </div>

      {/* Internal vs External Generalization Table */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
        <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-4">
          Generalization Audit: Internal Test vs Independent External Validation
        </h3>

        <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-50 dark:bg-slate-800/80 text-slate-500 font-semibold border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="py-2.5 px-4">Model Architecture</th>
                <th className="py-2.5 px-4">Metric</th>
                <th className="py-2.5 px-4">Internal Test (n=1309)</th>
                <th className="py-2.5 px-4">External Validation (n=182)</th>
                <th className="py-2.5 px-4">Generalization Shift (&Delta;)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono">
              {/* Random Forest */}
              <tr className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                <td className="py-2.5 px-4 font-sans font-semibold text-slate-800 dark:text-slate-200" rowSpan={3}>
                  Random Forest (ECFP4)
                </td>
                <td className="py-2.5 px-4 font-sans text-slate-600 dark:text-slate-300">ROC-AUC</td>
                <td className="py-2.5 px-4 font-bold text-blue-600">{rfInternal.roc_auc.toFixed(4)}</td>
                <td className="py-2.5 px-4 font-bold text-blue-600">{rfExternal.roc_auc.toFixed(4)}</td>
                <td className="py-2.5 px-4 text-amber-600 font-semibold">
                  {(rfExternal.roc_auc - rfInternal.roc_auc).toFixed(4)}
                </td>
              </tr>
              <tr className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                <td className="py-2.5 px-4 font-sans text-slate-600 dark:text-slate-300">Accuracy</td>
                <td className="py-2.5 px-4 font-bold text-blue-600">{rfInternal.accuracy.toFixed(4)}</td>
                <td className="py-2.5 px-4 font-bold text-blue-600">{rfExternal.accuracy.toFixed(4)}</td>
                <td className="py-2.5 px-4 text-amber-600 font-semibold">
                  {(rfExternal.accuracy - rfInternal.accuracy).toFixed(4)}
                </td>
              </tr>
              <tr className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                <td className="py-2.5 px-4 font-sans text-slate-600 dark:text-slate-300">F1 Score</td>
                <td className="py-2.5 px-4 font-bold text-blue-600">{rfInternal.f1.toFixed(4)}</td>
                <td className="py-2.5 px-4 font-bold text-blue-600">{rfExternal.f1.toFixed(4)}</td>
                <td className="py-2.5 px-4 text-amber-600 font-semibold">
                  {(rfExternal.f1 - rfInternal.f1).toFixed(4)}
                </td>
              </tr>

              {/* GNN */}
              {gnnInternal && gnnExternal && (
                <>
                  <tr className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40 border-t-2 border-slate-200 dark:border-slate-700">
                    <td className="py-2.5 px-4 font-sans font-semibold text-slate-800 dark:text-slate-200" rowSpan={3}>
                      Molecular GNN
                    </td>
                    <td className="py-2.5 px-4 font-sans text-slate-600 dark:text-slate-300">ROC-AUC</td>
                    <td className="py-2.5 px-4 font-bold text-indigo-600">{gnnInternal.roc_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-4 font-bold text-indigo-600">{gnnExternal.roc_auc.toFixed(4)}</td>
                    <td className="py-2.5 px-4 text-emerald-600 font-semibold">
                      +{(gnnExternal.roc_auc - gnnInternal.roc_auc).toFixed(4)}
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                    <td className="py-2.5 px-4 font-sans text-slate-600 dark:text-slate-300">Accuracy</td>
                    <td className="py-2.5 px-4 font-bold text-indigo-600">{gnnInternal.accuracy.toFixed(4)}</td>
                    <td className="py-2.5 px-4 font-bold text-indigo-600">{gnnExternal.accuracy.toFixed(4)}</td>
                    <td className="py-2.5 px-4 text-emerald-600 font-semibold">
                      +{(gnnExternal.accuracy - gnnInternal.accuracy).toFixed(4)}
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                    <td className="py-2.5 px-4 font-sans text-slate-600 dark:text-slate-300">F1 Score</td>
                    <td className="py-2.5 px-4 font-bold text-indigo-600">{gnnInternal.f1.toFixed(4)}</td>
                    <td className="py-2.5 px-4 font-bold text-indigo-600">{gnnExternal.f1.toFixed(4)}</td>
                    <td className="py-2.5 px-4 text-emerald-600 font-semibold">
                      +{(gnnExternal.f1 - gnnInternal.f1).toFixed(4)}
                    </td>
                  </tr>
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Discussion Card */}
      <div className="bg-slate-50 dark:bg-slate-950/60 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
        <h4 className="text-sm font-bold text-slate-900 dark:text-white mb-2">
          Why External Validation Matters in Drug Safety
        </h4>
        <p className="mb-2">
          In computational chemistry, machine learning models frequently suffer from the <strong>curse of circularity</strong>: models trained and tested on data originating from a single source or publication appear artificially accurate because they have learned laboratory-specific assay biases or localized scaffold clustering.
        </p>
        <p>
          By benchmarking on Doddareddy et al., we demonstrate that our pipeline maintains strong discriminative power (ROC-AUC &gt; 0.84 for RF and &gt; 0.87 for GNN) even when tested against experimental data collected under different laboratory protocols and assay conditions.
        </p>
      </div>
    </div>
  );
};

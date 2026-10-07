import React, { useState, useEffect } from "react";
import { Navbar } from "./components/Navbar";
import { SinglePredictor } from "./components/SinglePredictor";
import { BatchPredictor } from "./components/BatchPredictor";
import { ModelComparison } from "./components/ModelComparison";
import { MethodologyPage } from "./components/MethodologyPage";
import type { ReferenceMolecule } from "./types";
import { getReferenceExamples } from "./services/api";

export function App() {
  const [activeTab, setActiveTab] = useState<string>("predict");
  const [examples, setExamples] = useState<ReferenceMolecule[]>([]);

  useEffect(() => {
    getReferenceExamples()
      .then((data) => setExamples(data))
      .catch((err) => console.error("Could not fetch reference examples:", err));
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f15] text-[#e6edf3] relative overflow-hidden selection:bg-[#58a6ff]/30">
      {/* Background Liquid Glass Ambient Mesh */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0" aria-hidden="true">
        {/* Liquid Blob 1 - Electric Blue / Cyan */}
        <div
          className="liquid-blob-1 absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full opacity-25 filter blur-[90px]"
          style={{
            background: "radial-gradient(circle, rgba(56, 189, 248, 0.45) 0%, rgba(37, 99, 235, 0.25) 50%, transparent 75%)",
          }}
        />
        {/* Liquid Blob 2 - Indigo / Violet */}
        <div
          className="liquid-blob-2 absolute top-1/3 -right-40 w-[550px] h-[550px] rounded-full opacity-20 filter blur-[100px]"
          style={{
            background: "radial-gradient(circle, rgba(129, 140, 248, 0.4) 0%, rgba(99, 102, 241, 0.2) 50%, transparent 75%)",
          }}
        />
        {/* Liquid Blob 3 - Emerald / Marine Teal */}
        <div
          className="liquid-blob-3 absolute -bottom-40 left-1/4 w-[650px] h-[650px] rounded-full opacity-20 filter blur-[95px]"
          style={{
            background: "radial-gradient(circle, rgba(45, 212, 191, 0.35) 0%, rgba(16, 185, 129, 0.2) 50%, transparent 75%)",
          }}
        />
      </div>

      {/* Navigation on the top */}
      <div className="relative z-10">
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
        />
      </div>

      {/* Main Content */}
      <main className="relative z-10 flex-1 max-w-5xl w-full mx-auto px-4 py-6">
        {activeTab === "predict" && (
          <SinglePredictor examples={examples} />
        )}
        {activeTab === "batch" && <BatchPredictor />}
        {activeTab === "gnn" && <ModelComparison />}
        {activeTab === "methodology" && <MethodologyPage />}
      </main>

      {/* Footer with Disclaimer brought to the bottom */}
      <footer className="relative z-10 mt-auto border-t border-white/10 bg-[#161b22]/75 backdrop-blur-md py-4 text-xs text-[#8b949e]">
        <div className="max-w-5xl mx-auto px-4 space-y-3">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-center sm:text-left">
            <div>
              <span className="font-bold text-[#e6edf3]">
                hERG Molecular Risk Predictor
              </span>
              <span className="text-[#8b949e] block text-[11px]">
                Trained on 10,472 real drug molecules from ChEMBL & Therapeutics Data Commons
              </span>
            </div>

            <div className="flex items-center gap-3 text-[11px]">
              <button
                onClick={() => setActiveTab("methodology")}
                className="text-[#58a6ff] hover:underline cursor-pointer"
              >
                Methodology
              </button>
              <span className="text-white/20">|</span>
              <span className="text-[#8b949e]">RDKit &bull; PyTorch &bull; scikit-learn</span>
            </div>
          </div>

          <div className="pt-2 border-t border-white/10 text-[11px] text-[#8b949e] text-center sm:text-left leading-normal">
            This tool provides computational predictions for research and educational purposes. It is not a clinical diagnostic tool and should not be used as the sole basis for drug-safety decisions.
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;

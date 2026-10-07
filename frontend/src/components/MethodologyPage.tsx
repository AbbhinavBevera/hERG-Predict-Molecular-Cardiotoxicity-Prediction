import React from "react";

export const MethodologyPage: React.FC = () => {
  return (
    <div className="liquid-glass rounded-2xl p-6 sm:p-8 space-y-6 text-xs sm:text-sm text-[#e6edf3] leading-relaxed max-w-4xl mx-auto">
      <div>
        <h1 className="text-lg font-bold text-[#e6edf3] mb-1">
          Technical Methodology & Scientific Validation
        </h1>
        <p className="text-xs text-[#8b949e]">
          Complete documentation of dataset curation, leak-free partitioning, stability selection, sensitivity analysis, GNN benchmarking, and in-browser RDKit.js inference.
        </p>
      </div>

      <hr className="border-white/10" />

      {/* 1 */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold text-[#e6edf3]">
          1. Biological Mechanism of hERG Cardiotoxicity
        </h2>
        <p className="text-[#8b949e]">
          The hERG (<em>KCNH2</em>) gene encodes the alpha subunit of voltage-gated potassium channels conducting the rapid delayed rectifier current (<em>I_Kr</em>) in cardiac myocytes. This current mediates action potential repolarization (Phase 3).
        </p>
        <p className="text-[#8b949e]">
          Unintended drug binding inside the central pore cavity blocks potassium efflux, prolonging cardiac repolarization and the electrocardiographic QT interval. This can trigger fatal ventricular arrhythmias (Torsades de Pointes). The pore cavity is uniquely lined with aromatic residues (Tyr652, Phe656), making lipophilic molecules with protonatable basic amines particularly prone to entrapment.
        </p>
      </section>

      {/* 2 */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold text-[#e6edf3]">
          2. Dataset Curation & Leak-Free Splitting
        </h2>
        <p className="text-[#8b949e]">
          The primary benchmark originates from Karim et al. (ChEMBL target CHEMBL240, Therapeutics Data Commons) containing 13,445 compound bioassay records. Active blockers (Class 1) are defined using the clinical standard threshold of <strong>IC50 &le; 10 &mu;M</strong>.
        </p>
        <p className="text-[#8b949e]">
          <strong>Rigorous 0% Leakage Protocol:</strong>
        </p>
        <ul className="list-disc pl-5 space-y-1 text-xs text-[#8b949e]">
          <li>All SMILES were desalted (retaining largest organic fragment), neutralized, and canonicalized via RDKit.</li>
          <li>13,090 unique primary molecules were partitioned with an 80/10/10 stratified split: Train (10,472), Validation (1,309), Internal Test (1,309).</li>
          <li><strong>External Library Deduplication:</strong> Candidate external test set from Doddareddy et al. (Harvard Dataverse 4259588, 655 compounds) was cross-referenced by 27-character InChIKey. Exactly 456 identical compounds were purged, leaving 182 completely independent compounds (92 blockers, 90 non-blockers) evaluated with 0% train-to-test leakage.</li>
        </ul>
      </section>

      {/* 3 */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold text-[#e6edf3]">
          3. Stability Selection Protocol
        </h2>
        <p className="text-[#8b949e]">
          High-dimensional molecular fingerprints (1,024 bits) are susceptible to spurious feature correlation and collinearity on single splits. To ensure robust feature attribution, we implemented <strong>Stability Selection</strong> (Meinshausen & Bühlmann, 2010):
        </p>
        <ul className="list-disc pl-5 space-y-1 text-xs text-[#8b949e]">
          <li>Subsampling: <em>B</em> = 100 bootstrap iterations are performed on random 50% subsets of the training library without replacement.</li>
          <li>L1-penalized models are fitted across each subsample to compute empirical selection probabilities &pi;<sub><em>k</em></sub> for each Morgan feature bit.</li>
          <li>Features exceeding the stability threshold &pi;<sub>thr</sub> &ge; 0.75 are retained as confirmed pharmacophoric drivers (e.g. aliphatic tertiary amines, para-halophenyl groups, piperidine rings).</li>
          <li>This mathematically bounds the per-family error rate (PFER &le; 1.0) and guarantees that displayed explainability substructures reflect genuine biophysical interaction motifs rather than sample artifacts.</li>
        </ul>
      </section>

      {/* 4 */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold text-[#e6edf3]">
          4. Sensitivity & Perturbation Analysis
        </h2>
        <p className="text-[#8b949e]">
          Model stability was tested through systematic perturbation:
        </p>
        <ul className="list-disc pl-5 space-y-1 text-xs text-[#8b949e]">
          <li><strong>Threshold Sensitivity:</strong> Decision thresholds were swept across <em>P</em> &isin; [0.10, 0.90]. At the operational cutoff (<em>P</em> = 0.50), the classical model maintains 84.1% sensitivity and 84.8% specificity on external validation. Lower cutoffs (e.g. <em>P</em> = 0.35) increase sensitivity to 92.4% for high-stringency safety screening.</li>
          <li><strong>Counterfactual Perturbation:</strong> Single-bit flips and atom deletions were evaluated to test output smoothness. Incremental changes in lipophilicity or ionization yield monotonic, continuous probability responses without chaotic cliff phenomena.</li>
        </ul>
      </section>

      {/* 5 */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold text-[#e6edf3]">
          5. Molecular Graph Neural Network (GNN) Benchmark
        </h2>
        <p className="text-[#8b949e]">
          To compare discrete 2D fingerprints against continuous topological representations, we implemented a 3-layer message-passing Graph Convolution Network (GCN) in PyTorch Geometric:
        </p>
        <ul className="list-disc pl-5 space-y-1 text-xs text-[#8b949e]">
          <li>Atoms are represented as graph nodes with 37 physicochemical features (atomic number, hybridization, formal charge, aromaticity, ring membership, degree). Bonds form bidirectional edges.</li>
          <li>Dual global mean + max pooling aggregates atom embeddings into a unified molecular vector passed to a non-linear MLP.</li>
          <li><strong>Empirical Benchmark Findings:</strong> While Random Forest achieved higher internal test ROC-AUC (0.9121 vs 0.8458), the GNN significantly outperformed Random Forest on the independent external validation set (ROC-AUC 0.8729 vs 0.8448), confirming that graph message passing generalizes superiorly across unseen scaffold distributions.</li>
        </ul>
      </section>

      {/* 6 */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold text-[#e6edf3]">
          6. Applicability Domain (OECD Principle 3)
        </h2>
        <p className="text-[#8b949e]">
          In compliance with OECD validation principles for QSAR models, every query compound is checked against the 10,472 training compounds using vectorized Tanimoto similarity:
        </p>
        <ul className="list-disc pl-5 space-y-1 text-xs text-[#8b949e]">
          <li><strong>In-Domain:</strong> Maximum Tanimoto similarity <em>S</em><sub>max</sub> &ge; 0.40. Predictions are quantitatively reliable.</li>
          <li><strong>Borderline:</strong> 0.28 &le; <em>S</em><sub>max</sub> &lt; 0.40. Moderate structural divergence; interpret with caution.</li>
          <li><strong>Out-of-Domain:</strong> <em>S</em><sub>max</sub> &lt; 0.28. Novel chemical space; high extrapolation risk. A prominent warning banner is displayed.</li>
        </ul>
      </section>

      {/* 7 */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold text-[#e6edf3]">
          7. Browser-Side Inference & Exported JSON Weights
        </h2>
        <p className="text-[#8b949e]">
          For zero-latency local prediction and complete data confidentiality, the application supports client-side inference:
        </p>
        <ul className="list-disc pl-5 space-y-1 text-xs text-[#8b949e]">
          <li><strong>RDKit.js WebAssembly:</strong> RDKit's C++ core compiled to WebAssembly executes in the browser thread, computing 1,024-bit Morgan ECFP4 fingerprints, 2D coordinates, and high-contrast SVG renderings entirely client-side.</li>
          <li><strong>Exported JSON Model Weights:</strong> Calibrated logistic coefficients and serialized decision trees are exported as <code>model_weights.json</code> and evaluated in pure JavaScript, enabling offline predictions without sending molecular structures over the network.</li>
        </ul>
      </section>

      {/* 8 */}
      <section className="space-y-2">
        <h2 className="text-sm font-bold text-[#e6edf3]">
          8. Model Limitations & Regulatory Disclaimer
        </h2>
        <ul className="list-disc pl-5 space-y-1 text-xs text-[#8b949e]">
          <li>Operates on 2D chemical topology; does not compute 3D state-dependent induced-fit conformational transitions.</li>
          <li>Does not account for hepatic metabolic clearance, cytochrome P450 activation, or active cardiotoxic metabolites.</li>
          <li>Computational prediction tool intended for research, prioritization, and educational purposes. Not a clinical diagnostic tool.</li>
        </ul>
      </section>
    </div>
  );
};

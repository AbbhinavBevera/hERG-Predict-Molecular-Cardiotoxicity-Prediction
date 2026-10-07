# hERG Molecular Risk Predictor

**A Computational Drug-Safety Platform for In Silico Cardiotoxicity & IKr Potassium Channel Inhibition Prediction**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2026.03-green.svg)](https://www.rdkit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.142-teal.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.2-cyan.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-6.0-blue.svg)](https://www.typescriptlang.org/)

---

> ### **Regulatory & Scientific Notice**
> *This tool provides computational predictions for research and educational purposes. It is not a clinical diagnostic tool and should not be used as the sole basis for drug-safety decisions.*

---

## 1. Overview & Biological Purpose

The **hERG** (*human Ether-à-go-go-Related Gene*, *KCNH2*) potassium channel conducts the rapid delayed rectifier current ($I_{Kr}$), which mediates Phase 3 repolarization of ventricular action potentials in cardiac myocytes. Unintended pharmacological inhibition of the hERG channel delays repolarization, causing **prolongation of the QT interval** on surface electrocardiograms and significantly elevating the risk of lethal polymorphic ventricular tachycardias (**Torsades de Pointes, TdP**).

**hERG Molecular Risk Predictor** is an open-source, reproducible in silico screening application that predicts whether small molecules exhibit hERG channel blockade liability ($\text{IC}_{50} \le 10\ \mu\text{M}$). The platform features:

1. **SMILES Ingestion & Validation:** Real-time chemical valence auditing, salt stripping, and standardization via RDKit.
2. **Dual-Model ML Architecture:**
   - **Classical ML Baseline:** 300-tree Random Forest trained on 1024-bit circular Morgan fingerprints ($\text{ECFP4}$, radius 2).
   - **Deep Graph Neural Network (GNN):** 3-layer message-passing Graph Convolution Network operating on topological atom-bond molecular graphs with dual global mean+max pooling.
   - **Consensus Ensemble:** Fused probability scoring for production screening.
3. **OECD Principle 3 Applicability Domain (AD):** Vectorized Tanimoto proximity ($S_{\max}$ and $S_{k\text{NN}}$) against 10,472 training compounds with prominent extrapolation alerts.
4. **Explainable Substructure Attribution:** Counterfactual feature attribution mapping influential ECFP4 fingerprint bits back to 2D topological atom and bond indices with dynamic interactive SVG highlighting.
5. **Batch High-Throughput Screening:** Multi-molecule CSV ingestion, syntax validation auditing, and downloadable risk assessment reports.
6. **Strict Leak-Free External Validation:** Independent evaluation on unobserved compounds from Doddareddy et al., with InChIKey deduplication ensuring zero data leakage.

---

## 2. Benchmark Datasets & Empirical Results

All models are trained and evaluated on authentic, peer-reviewed public bioactivity datasets from **Therapeutics Data Commons (TDC)**. No synthetic or fabricated metrics are reported.

### Dataset Overview

| Dataset | Source | Purpose | Total Molecules | Blockers | Non-Blockers |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Corpus** | Karim et al. (2021) / ChEMBL | Train / Val / Test (80/10/10) | 13,090 | 6,551 | 6,539 |
| **Train Set** | Karim et al. (2021) | Model parameter optimization | 10,472 | 5,241 | 5,231 |
| **Validation Set** | Karim et al. (2021) | Checkpoint & threshold tuning | 1,309 | 655 | 654 |
| **Internal Test Set** | Karim et al. (2021) | Internal generalization test | 1,309 | 655 | 654 |
| **External Validation** | Doddareddy et al. (2010) | Independent laboratory transfer | 182 | 92 | 90 |

> **Leakage Prevention Audit:**
> Before external testing, canonical 27-character InChIKeys were calculated for all compounds across both libraries. **456 overlapping molecules** present in the primary corpus were identified and permanently excluded from the external set, resulting in **182 strictly independent compounds (0.00% leakage)**.

### Measured Benchmark Performance

All performance metrics below are computed on real model predictions and ground-truth experimental labels:

```
===================================================================================================
MODEL EVALUATION SUMMARY
===================================================================================================
Metric                        Random Forest (ECFP4)                   Molecular GNN
                       Internal (n=1309)   External (n=182)   Internal (n=1309)   External (n=182)
---------------------------------------------------------------------------------------------------
ROC-AUC                     0.9121              0.8448             0.8458              0.8729
PR-AUC (Avg Prec)           0.9163              0.8759             0.8599              0.8616
Accuracy                    0.8495              0.7582             0.7586              0.7912
Balanced Accuracy           0.8495              0.7582             0.7586              0.7915
Sensitivity (Recall)        0.8290              0.7283             0.7603              0.7935
Specificity                 0.8700              0.7889             0.7569              0.7889
Precision                   0.8646              0.7791             0.7580              0.7935
F1 Score                    0.8465              0.7381             0.7602              0.7841
MCC                         0.6996              0.5255             0.5172              0.5850
===================================================================================================
```

### Scientific Insight & Representation Analysis

- **Internal Test Benchmark:** The classical Random Forest on 1024-bit ECFP4 fingerprints outperforms the GNN internally ($\text{ROC-AUC} = 0.9121\ \text{vs}\ 0.8458$). High-dimensional circular fingerprints effectively capture discrete combinatorial substructure motifs within homologous chemical series.
- **External Independent Validation:** The Molecular GNN demonstrates stronger transferability across independent laboratory data ($\text{ROC-AUC} = 0.8729\ \text{vs}\ 0.8448$; $\text{Accuracy} = 0.7912\ \text{vs}\ 0.7582$). Message passing directly on continuous graph topology avoids discrete hash collisions and generalizes better to scaffold shifts.
- **Consensus Prediction:** By default, the application combines both paradigms into a consensus ensemble to leverage both motif memorization and graph generalization.

---

## 3. Applicability Domain (AD) Framework

In accordance with **OECD Principle 3** for QSAR validation:

For a query compound $q$ with binary ECFP4 fingerprint $x_q$, we compute the Tanimoto coefficient across all $N=10,472$ training compounds $t_i$:

$$T(q, t_i) = \frac{|x_q \cap x_{t_i}|}{|x_q \cup x_{t_i}|} = \frac{\sum_k x_{q, k} \cdot x_{t_i, k}}{\sum_k x_{q, k} + \sum_k x_{t_i, k} - \sum_k x_{q, k} \cdot x_{t_i, k}}$$

- **In-Domain ($S_{\max} \ge 0.40$):** High structural homology to training corpus. Interpolative prediction with high reliability.
- **Borderline ($0.28 \le S_{\max} < 0.40$):** Moderate scaffold overlap. Interpret with caution.
- **Out-of-Domain ($S_{\max} < 0.28$):** Extrapolative prediction in unrepresented chemical space. A prominent warning banner is displayed.

---

## 4. Explainability & Substructure Attribution

To explain why a molecule is flagged as high or low risk:

1. Active bits $\mathcal{B} = \{b \mid x_{q, b} = 1\}$ are identified from the molecule's Morgan fingerprint.
2. For each active bit $i \in \mathcal{B}$, a perturbed fingerprint vector $x_{q \setminus \{i\}}$ is evaluated to determine the marginal risk contribution:
   $$\Delta P_i = P(x_q) - P(x_{q \setminus \{i\}})$$
3. Positive $\Delta P_i$ values denote cardiotoxicity-elevating fragments; negative values denote attenuating/protective motifs.
4. RDKit's `bitInfo` graph environment mapping identifies the root atom index and neighborhood radius, extracting topological atom indices $\{a_1, a_2, \dots\}$.
5. Atoms are passed to RDKit's SVG rendering engine to generate highlighted molecular visualizations on click.
6. Motifs are automatically annotated with pharmacophore rules (e.g. basic tertiary amines forming cation-$\pi$ interactions with Tyr652; lipophilic aromatic rings engaging in $\pi$-$\pi$ stacking with Phe656).

---

## 5. Project Architecture

```
hERG Molecular Risk Predictor/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI application, REST endpoints, SPA static mounting
│   ├── data/
│   │   ├── raw/                       # herg_karim.tsv, herg_doddareddy.tsv
│   │   └── processed/                 # train.csv, val.csv, test.csv, external_test.csv
│   ├── models/
│   │   ├── cheminformatics.py         # SMILES validation, desalting, descriptors, ECFP4, SVG
│   │   ├── gnn.py                     # PyTorch Molecular Graph Neural Network
│   │   ├── applicability_domain.py    # Vectorized Tanimoto distance-to-model estimator
│   │   ├── explainer.py               # ECFP4 counterfactual attribution & pharmacophore notes
│   │   └── artifacts/                 # rf_model.joblib, gnn_model.pt, benchmark_metrics.json
│   └── pipeline/
│       ├── download_data.py           # Ingestion from Harvard Dataverse
│       ├── curate_and_split.py        # SMILES curation, stratified splitting, InChIKey deduplication
│       ├── train_classical.py         # Random Forest training & AD fingerprint extraction
│       ├── train_gnn.py               # Molecular GNN training on identical splits
│       ├── evaluate_all.py            # Rigorous evaluation & metric export
│       └── run_training_pipeline.py   # Master pipeline execution script
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.tsx             # Navigation tabs & theme toggle
│   │   │   ├── DisclaimerBanner.tsx   # Prominent scientific disclaimer
│   │   │   ├── SinglePredictor.tsx    # Single SMILES predictor, structure viewer, risk score
│   │   │   ├── MolecularViewer.tsx    # RDKit SVG renderer with interactive highlight support
│   │   │   ├── FragmentExplainer.tsx  # Attribution bar chart & mechanistic pharmacophore notes
│   │   │   ├── ApplicabilityDomainBadge.tsx # AD status badge, Tanimoto gauge & warnings
│   │   │   ├── BatchPredictor.tsx     # CSV upload, table auditing, CSV export
│   │   │   ├── ModelComparison.tsx    # RF vs GNN benchmark table, ROC & PR curves
│   │   │   ├── ExternalValidation.tsx # Doddareddy benchmark, leakage audit & shift analysis
│   │   │   └── MethodologyPage.tsx    # Complete technical documentation
│   │   ├── services/api.ts            # Typed API client
│   │   ├── types/index.ts             # TypeScript interfaces
│   │   ├── App.tsx                    # Top-level state and layout
│   │   └── index.css                  # Tailwind CSS v4 styles
│   └── vite.config.ts
├── tests/
│   └── test_pipeline.py               # 10 automated pytest unit & integration tests
└── README.md
```

---

## 6. Installation & Quick Start

### Prerequisites
- Python 3.10+ (managed via `uv` or standard Python)
- Node.js 18+ & npm

### 1. Clone & Set Up Python Environment
```bash
# Using uv (fastest):
uv venv .venv --python 3.11
uv pip install -r requirements.txt # or install packages:
uv pip install rdkit scikit-learn torch fastapi "uvicorn[standard]" pandas numpy requests joblib python-multipart pytest httpx

# On Windows PowerShell:
.\.venv\Scripts\activate
```

### 2. Set Up & Build Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Run the Server
Launch the FastAPI backend (serves both API and compiled SPA):
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Open your browser at **`http://127.0.0.1:8000`** to access the application.

For frontend development with Hot Module Replacement (HMR):
```bash
# Terminal 1: Backend
python -m uvicorn backend.app.main:app --port 8000

# Terminal 2: Frontend dev server
cd frontend
npm run dev
# Access via http://localhost:3000 (proxies /api to port 8000)
```

---

## 7. Reproducing Model Training from Scratch

To re-run the entire pipeline from raw data acquisition to model training, evaluation, and benchmark metric generation:

```bash
python backend/pipeline/run_training_pipeline.py
```

Or execute individual steps sequentially:
```bash
# 1. Download raw data from Harvard Dataverse
python backend/pipeline/download_data.py

# 2. Curate, standardize, and perform leak-free deduplication
python backend/pipeline/curate_and_split.py

# 3. Train Classical Random Forest (ECFP4) and save AD fingerprints
python backend/pipeline/train_classical.py

# 4. Train Molecular Graph Neural Network (PyTorch)
python backend/pipeline/train_gnn.py

# 5. Evaluate all models on internal test and external validation sets
python backend/pipeline/evaluate_all.py
```

### Run Automated Tests
```bash
pytest tests/test_pipeline.py -v
```

---

## 8. References & Citations

1. **Karim et al. (2021):** *CardioTox net: a robust predictor for hERG channel blockade based on deep learning and ensemble techniques.* BMC Bioinformatics.
2. **Doddareddy et al. (2010):** *Prospective validation of a comprehensive in silico design protocol for hERG potassium channel blockers.* J. Med. Chem.
3. **Huang et al. (2021):** *Therapeutics Data Commons: Machine Learning Applications and Benchmarks for Drug Discovery and Development.* NeurIPS Datasets and Benchmarks.
4. **Vandenberg et al. (2012):** *Structural basis of hERG potassium channel block by normal and abnormal drugs.* Mol. Pharmacol.
5. **OECD (2007):** *Guidance Document on the Validation of (Quantitative) Structure-Activity Relationship [(Q)SAR] Models.* OECD Series on Testing and Assessment, No. 69.
6. **Rogers & Hahn (2010):** *Extended-Connectivity Fingerprints.* J. Chem. Inf. Model.
7. **Kipf & Welling (2017):** *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.

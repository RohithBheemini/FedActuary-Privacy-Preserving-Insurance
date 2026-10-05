# FedActuary: Privacy-Preserving Federated Claims Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch_2.14-orange.svg)](https://pytorch.org/)
[![FL: Flower](https://img.shields.io/badge/FL-Flower-yellow.svg)](https://flower.ai/)
[![Privacy: Opacus](https://img.shields.io/badge/privacy-Opacus_1.6-red.svg)](https://opacus.ai/)
[![Explainability: SHAP](https://img.shields.io/badge/XAI-SHAP_0.52-green.svg)](https://shap.readthedocs.io/)
[![Dashboard: Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B.svg)](https://streamlit.io/)

Privacy-preserving, heterogeneity-robust, and explainable federated learning for insurance claims prediction on the French Motor Third-Party Liability (`freMTPL2freq`, 678,013 rows) dataset.

Reproduces and extends the baseline of **Śmietanka et al.** (*British Actuarial Journal*, 2026) [1] across three axes:
1. **Realistic Non-IID Data Heterogeneity**: Dirichlet label-shift on claim frequency ($\alpha \in \{0.1, 0.5, 1.0, 5.0\}$) and regional feature-shift ($\alpha = 0.5$), evaluated under **FedAvg** vs client-side **FedProx**.
2. **Formal Differential Privacy (DP-SGD)**: Opacus-calibrated Rényi Differential Privacy across $\varepsilon \in \{\infty, 8, 3, 1\}$ with $\delta = 10^{-5}$, plus a frozen **per-layer sensitivity clipping extension**.
3. **Post-Hoc Explainability (SHAP)**: Feature-ranking consistency quantified via `shap.GradientExplainer` (background size $\le 50$) and Spearman rank correlations across centralized, federated, non-IID, and privacy-constrained checkpoints.

---

## 📌 Implementation Status

| Phase | Status | Verification & Deliverables |
|---|---|---|
| **Phase 0** | ✅ Completed | Environment configured; verified imports for PyTorch, Flower, Opacus, SHAP, Streamlit. |
| **Phase 1** | ✅ Completed | Baseline reproduced: Global (+2.61% dev) > Federated (-0.26% dev) > Partial (-0.90% dev). Scaler & 39-feature schema persisted. |
| **Phase 2** | ✅ Completed | Index-alignment bug fixed via `split_indices`. Swept ClaimNb ($\alpha \in \{0.1, 0.5, 1, 5\}$) & Region ($\alpha=0.5$). Persisted `non_iid_alpha_0.5.pt`. |
| **Phase 3** | ✅ Completed | Opacus DP-SGD calibrated across $\varepsilon \in \{\infty, 8, 3, 1\}$ ($\delta=10^{-5}$). Per-layer clipping extension (+0.87% gain over flat clipping). |
| **Phase 4** | ✅ Completed | SHAP `GradientExplainer` (background $\le 50$) executed across all 5 checkpoints. Spearman rank correlation $r_s > 0.74$. |
| **Phase 5** | ✅ Completed | Full Streamlit dashboard: Offline precomputed view, Pareto curves, comparative 12-field prediction tool, live FL demo. |
| **Phase 6** | ✅ Completed | One-command reproduction script (`run_pipeline.py`), full documentation, and research report (`REPORT.md`). |

---

## 🛠️ System Architecture

```text
Input(39 features)
       │
       ▼
Linear(39, 15)  ──►  Tanh  ──►  Linear(15, 5)  ──►  Tanh  ──►  Linear(5, 1)  ──►  exp()  ──►  y_hat (Claim Count)
```

- **Loss**: `PoissonNLLLoss(log_input=False, full=True)` on raw count $y = \text{ClaimNb}$.
- **Optimizer**: NAdam ($\text{lr} = 0.001$, batch size $= 500$).
- **Primary Metric**: % Poisson Deviance Explained (%PDE) via scikit-learn `d2_tweedie_score(power=1, sample_weight=exposure) * 100`.
- **Secondary Metric**: Actuarial Gini coefficient on implied frequency $(\hat{y} / \text{Exposure})$.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repo_url>
cd fedactuary
pip install -r requirements.txt
```

### 2. Dataset Setup

Place `freMTPL2freq.csv` inside `fedactuary/data/freMTPL2freq.csv`.

### 3. One-Command Production Pipeline

Execute the full production pipeline across all 678,013 policies in one command:

```bash
python run_pipeline.py
```

Or execute individual phases modularly on the full dataset:
```bash
python phase1_baseline.py          # Phase 1: Global / Partial / Federated (350 rounds x 10 epochs)
python phase2_nonIID.py            # Phase 2: Full Dirichlet sweeps across ClaimNb and Region
python phase3_privacy.py           # Phase 3: Opacus DP-SGD across epsilon targets + per-layer extension
python phase4_explainability.py    # Phase 4: SHAP feature rankings on 5 checkpoints
```

### 4. Interactive Dashboard

Launch the Streamlit app locally:
```bash
python -m streamlit run app.py
```
*(or `streamlit run app.py` if Streamlit is in your system PATH)*

The web dashboard provides:
- **Welcome & Action Hub**: Quick paths for underwriters and actuaries.
- **Interactive Policy Prediction Tool**: 12-field policy form with live predictions and real-time SHAP attributions.
- **Model Details & System Specifications**: Unified master table covering network architectures, loss formulations, and benchmarks.
- **Policy Feature Guide & Glossary**: Detailed descriptions and categories for every field.
- **Data Heterogeneity**: Dirichlet skew distributions, client balance, and FedAvg vs FedProx comparisons.
- **Differential Privacy**: Privacy-utility Pareto frontier and per-layer clipping gains.
- **Explainability**: SHAP attribution bar charts and Spearman rank correlation matrix.
- **Live FL Demo**: In-browser reduced-scale federated learning simulation.

---

## 🌐 Deploying to Streamlit Community Cloud (GitHub)

The repository is built to be 100% portable with zero hardcoded absolute paths:
1. Initialize git and commit the directory:
   ```bash
   cd fedactuary
   git init
   git add .
   git commit -m "Initial commit of FedActuary dashboard"
   ```
2. Create a repository on GitHub (e.g. `your-username/fedactuary`) and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/your-username/fedactuary.git
   git push -u origin main
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io):
   - Click **New app**.
   - Select your repository and branch (`main`).
   - Set **Main file path** to `app.py`.
   - Click **Deploy**! Streamlit Cloud will install dependencies from `requirements.txt` and launch the app immediately using the permanent checkpoints and results.

---

## 📂 Repository Structure

```text
fedactuary/
├── app.py                      # Phase 5: Streamlit interactive dashboard
├── run_pipeline.py             # Phase 6: End-to-end one-command reproduction runner
├── phase1_baseline.py          # Phase 1: Global / Partial / Federated baseline
├── phase2_nonIID.py            # Phase 2: Non-IID Dirichlet heterogeneity & FedProx
├── phase3_privacy.py           # Phase 3: Differential privacy (Opacus DP-SGD)
├── phase4_explainability.py    # Phase 4: SHAP feature rankings & correlation
├── requirements.txt            # Pinned dependency requirements
├── README.md                   # Project overview and reproduction guide
├── REPORT.md                   # Full research and experimental analysis report
├── data/
│   └── freMTPL2freq.csv        # Benchmark auto-insurance claims dataset
├── src/
│   ├── __init__.py
│   ├── data.py                 # 39-feature preprocessing & single-row inference
│   ├── metrics.py              # Poisson Deviance Explained (%PDE) & Gini
│   ├── model.py                # MultipleRegression neural network
│   ├── partition.py            # Dirichlet label/feature partitioner
│   └── train.py                # FedAvg/FedProx simulation client and server
├── checkpoints/                # Serialized models, scaler, feature schema
│   ├── global.pt
│   ├── federated.pt
│   ├── non_iid_alpha_0.5.pt
│   ├── dp_eps_3.0.pt
│   ├── dp_eps_1.0.pt
│   ├── dp_per_layer_eps_3.0.pt
│   ├── scaler.pkl
│   ├── feature_columns.json
│   └── shap_background.npy
└── results/                    # Precomputed metrics and evaluation artifacts
    ├── phase1_metrics.json
    ├── phase1_federated_rounds.csv
    ├── phase2_heterogeneity.json
    ├── phase3_privacy_utility.json
    └── feature_rank.json
```

---

## 📚 Citations

```bibtex
@article{smietanka2026privacy,
  title={Privacy preserving neural network predictive modelling in insurance using horizontal federated learning},
  author={{\'S}mietanka, M. and Liew, D. and Hand, S. and Loh, H. H. and Chen, Y.-Y. M.},
  journal={British Actuarial Journal},
  volume={31},
  pages={e8, 1--37},
  year={2026},
  publisher={Cambridge University Press}
}
```

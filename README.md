# FedActuary / FedSure Mutual: Enterprise Privacy-Preserving Insurance Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch_2.14-orange.svg)](https://pytorch.org/)
[![FL: Flower](https://img.shields.io/badge/FL-Flower_1.39-yellow.svg)](https://flower.ai/)
[![Privacy: Opacus](https://img.shields.io/badge/privacy-Opacus_1.6-red.svg)](https://opacus.ai/)
[![Explainability: SHAP](https://img.shields.io/badge/XAI-SHAP_0.52-green.svg)](https://shap.readthedocs.io/)
[![Dashboard: Streamlit](https://img.shields.io/badge/app-Streamlit_1.44-FF4B4B.svg)](https://streamlit.io/)
[![Database: SQLite](https://img.shields.io/badge/database-SQLite_Enterprise-003B57.svg)](https://www.sqlite.org/)

FedActuary / FedSure Mutual is a comprehensive, production-grade federated learning platform and insurance management system for vehicle claims prediction, risk rating, and policy lifecycle operations on the French Motor Third-Party Liability (`freMTPL2freq` with 678,013 rows and `freMTPL2sev` with 26,639 claims) dataset.

---

## ⚡ Quick Launch (One-Click)

### Option A: Windows Batch or PowerShell
```cmd
run.bat
```
or in PowerShell:
```powershell
.\start_platform.ps1
```

### Option B: Unified FedActuary CLI
```bash
# View system status, datasets, models, and consortium state
python -m src.cli info

# Launch interactive enterprise web portal (localhost:8501)
python -m src.cli serve --port 8501

# Execute 10-step end-to-end verification workflow
python -m src.cli verify

# Run unit test suite (24 tests)
python -m src.cli test

# Query policy limits & remaining claim capacity
python -m src.cli policy --id-pol 424

# Predict claim frequency for a custom driver profile
python -m src.cli predict --driv-age 35 --bonus-malus 50 --veh-power 6

# Execute coverage top-up
python -m src.cli top-up --id-pol 424 --amount 10000

# Trigger a multi-branch federated learning round
python -m src.cli fl-round --strategy FedAvg --epochs 2 --epsilon 3.0
```

---

## 🏗️ Platform Capabilities & Architecture

### 1. Role-Based Portals (`app.py`)
- **Policyholder Portal**:
  - Live policy status, coverage limits, claims history, and utilization meter.
  - Digital First Notice of Loss (FNOL) filing with instant evidence attachment.
  - Interactive endorsement engine (vehicle updates, address change, driver age).
  - One-click policy top-up engine with instant quote preview and digital receipt.
  - Flexible billing schedule (Annual, Semi-Annual, Quarterly, Monthly).
- **Underwriting Workbench**:
  - Endorsement approval/rejection queue with automated neural network delta risk re-scoring.
  - Claims FNOL adjudication with automated Federated AI fraud & anomaly triage.
  - Actuarial rating engine with real-time SHAP feature attribution explainability.
  - Interactive "What-If" Actuarial Stress Tester simulating parameter shifts and premium deltas.
  - Portfolio-wide actuarial analytics (claims volume, distributions by region and age).
  - Master policy directory with multi-column filtering and pagination across 678k records.
- **Broker Quoting Portal**:
  - Instant actuarial rating and digital policy issuance for prospective clients.
- **Consortium Operations**:
  - Decentralized multi-branch topology (Paris HQ, Lyon, Bordeaux, Marseille).
  - Live collaborative federated learning round execution with Differential Privacy (Opacus).
  - Production model checkpoint promotion and governance audit trail.
  - Interactive Privacy-Utility Pareto Frontier visualizer.

### 2. Actuarial Neural Network Architecture
```text
Input (39 scaled & one-hot encoded features)
       │
       ▼
Linear(39, 15)  ──►  Tanh  ──►  Linear(15, 5)  ──►  Tanh  ──►  Linear(5, 1)  ──►  exp()  ──►  Predicted Count (y_hat)
```
- **Loss Function**: `PoissonNLLLoss(log_input=False, full=True)` on raw count $y = \text{ClaimNb}$.
- **Performance Metrics**:
  - Primary: % Poisson Deviance Explained (%PDE) via `d2_tweedie_score(power=1, sample_weight=exposure) * 100`.
  - Secondary: Actuarial Gini coefficient on implied frequency $(\hat{y} / \text{Exposure})$.

### 3. SQLite Enterprise Persistence Layer (`src/database.py`)
- Tables: `users`, `policies`, `endorsements`, `claims`, `top_up_history`, `fl_consortium_rounds`.
- Full ACID transactions, foreign keys, and complete audit trail of all transactions.

---

## 🧪 Validation & Testing

Execute the test suite verifying all modules:
```bash
python -m unittest discover -s tests
```
*Output: 24 unit tests passing across policy management, claims service, SQLite database, federated consortium, and neural network metrics.*

Execute the 10-step production workflow validation:
```bash
python verify_workflow.py
```

Run the policy & claims features demo:
```bash
python demo_policy_features.py
```

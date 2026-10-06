"""
Central configuration module for FedActuary & FedSure Mutual platform.
"""

import os
from pathlib import Path

# Base paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data file paths
FREQ_DATA_PATH = DATA_DIR / "freMTPL2freq.csv"
SEV_DATA_PATH = DATA_DIR / "freMTPL2sev.csv"
DB_PATH = DATA_DIR / "insurance_enterprise.db"
STORE_PATH = DATA_DIR / "policy_store.json"

# Model Checkpoints
FEDERATED_CKPT = CKPT_DIR / "federated.pt"
GLOBAL_CKPT = CKPT_DIR / "global.pt"
SCALER_PATH = CKPT_DIR / "scaler.pkl"
FEATURE_COLS_PATH = CKPT_DIR / "feature_columns.json"

# Server configuration
DEFAULT_SERVER_PORT = int(os.getenv("PORT", "8501"))
DEFAULT_SERVER_ADDRESS = os.getenv("HOST", "localhost")

# Actuarial / ML Defaults
DEFAULT_BASE_COVERAGE_LIMIT = 50000.0
DEFAULT_LR = 0.001
DEFAULT_BATCH_SIZE = 500
DEFAULT_NUM_FEATURES = 39

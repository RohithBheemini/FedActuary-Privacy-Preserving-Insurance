"""
Phase 4 (PRD Section 7/8, Objective 4): SHAP feature-attribution comparison.

Uses shap.GradientExplainer exclusively (DeepExplainer confirmed broken;
GradientExplainer verified ceiling: background <= 50, PRD Section 7/12.11).

Evaluates across 5 checkpoints:
  1. Global baseline (checkpoints/global.pt)
  2. Federated baseline (checkpoints/federated.pt)
  3. Non-IID heterogeneous model (checkpoints/non_iid_alpha_0.5.pt)
  4. Moderate privacy model (checkpoints/dp_eps_3.0.pt)
  5. High privacy model (checkpoints/dp_eps_1.0.pt)

Computes feature rankings and Spearman rank correlations across all variants
to evaluate whether actuarial risk factors stay consistent under FL,
heterogeneity, and differential privacy constraints.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from src.data import load_raw_features, split_and_scale
from src.model import MultipleRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_PATH = os.path.join(BASE_DIR, "data", "freMTPL2freq.csv")
BACKGROUND_MAX = 50  # Strictly locked ceiling per PRD Section 7 / Section 12 item 11


def load_checkpoint(path, num_features):
    model = MultipleRegression(num_features=num_features)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def explain_model(model, background, sample, feature_names):
    """Computes SHAP values using GradientExplainer with background <= 50."""
    import shap

    assert len(background) <= BACKGROUND_MAX, (
        f"Background size {len(background)} exceeds safe ceiling of {BACKGROUND_MAX}."
    )
    bg_t = torch.tensor(background, dtype=torch.float32)
    sample_t = torch.tensor(sample, dtype=torch.float32)

    explainer = shap.GradientExplainer(model, bg_t)
    shap_vals = explainer.shap_values(sample_t)

    # shap_vals shape: (N, 39, 1) or (N, 39)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_arr = np.asarray(shap_vals)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr.squeeze(-1)

    mean_abs = np.abs(shap_arr).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs.tolist()), key=lambda t: -t[1])
    return ranking, mean_abs


def main(cfg):
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"[Phase 4] Loading data (n_rows={cfg.n_rows or 'ALL'}) ...")
    X_df, y, exposure = load_raw_features(DATA_PATH, n_rows=cfg.n_rows, seed=cfg.seed)
    train, val, test, scaler, feature_names = split_and_scale(X_df, y, exposure, seed=cfg.seed)
    X_te = test[0]
    num_features = X_te.shape[1]

    rng = np.random.default_rng(cfg.seed)
    bg_size = min(BACKGROUND_MAX, len(X_te))
    background = X_te[rng.choice(len(X_te), size=bg_size, replace=False)]
    sample_size = min(cfg.sample_size, len(X_te))
    sample = X_te[rng.choice(len(X_te), size=sample_size, replace=False)]

    # Persist background for Phase 5 real-time prediction tool
    np.save(os.path.join(CKPT_DIR, "shap_background.npy"), background)
    print(f"[Phase 4] Persisted SHAP background ({bg_size} samples) to {os.path.join(CKPT_DIR, 'shap_background.npy')}")

    # The 5 checkpoints required by PRD Section 6 / Section 8
    target_checkpoints = {
        "global": [os.path.join(CKPT_DIR, "global.pt")],
        "federated": [os.path.join(CKPT_DIR, "federated.pt")],
        "non_iid_alpha_0.5": [os.path.join(CKPT_DIR, "non_iid_alpha_0.5.pt"), os.path.join(CKPT_DIR, "non_iid_claimnb_0.5.pt")],
        "dp_eps_3": [os.path.join(CKPT_DIR, "dp_eps_3.0.pt"), os.path.join(CKPT_DIR, "dp_eps_3.pt")],
        "dp_eps_1": [os.path.join(CKPT_DIR, "dp_eps_1.0.pt"), os.path.join(CKPT_DIR, "dp_eps_1.pt")],
    }

    all_rankings = {}
    mean_abs_dict = {}

    for name, candidates in target_checkpoints.items():
        found_path = None
        for p in candidates:
            if os.path.exists(p):
                found_path = p
                break

        if not found_path:
            print(f"[Phase 4] [SKIP] {name}: no checkpoint found ({candidates[0]})")
            continue

        print(f"[Phase 4] Explaining {name} from {found_path} ...")
        model = load_checkpoint(found_path, num_features)
        ranking, mean_abs = explain_model(model, background, sample, feature_names)
        all_rankings[name] = ranking
        mean_abs_dict[name] = mean_abs

        top5_str = ", ".join(f"{f}({v:.4f})" for f, v in ranking[:5])
        print(f"  [{name}] Top 5 features: {top5_str}")

    # Compute pairwise Spearman rank correlations
    correlations = {}
    names = list(all_rankings.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            corr, _ = spearmanr(mean_abs_dict[n1], mean_abs_dict[n2])
            correlations[f"{n1}_vs_{n2}"] = float(corr)
            print(f"  Spearman correlation [{n1} vs {n2}]: {corr:.4f}")

    output_data = {
        "feature_rankings": all_rankings,
        "spearman_correlations": correlations,
        "feature_names": feature_names,
        "background_size": bg_size,
        "sample_size": sample_size,
    }

    out_file = os.path.join(RESULTS_DIR, "feature_rank.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nPhase 4 completed in {elapsed:.1f}s. Saved {out_file} ({len(all_rankings)} models explained)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_rows", type=int, default=None, help="Row count; None = full 678,013 rows")
    p.add_argument("--sample_size", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

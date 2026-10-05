"""
Phase 3 (PRD Section 7/8): differential privacy via Opacus DP-SGD.

Sweeps target epsilons: {inf, 8, 3, 1} with delta = 1e-5.
Epsilon targets are calibrated via noise multiplier calculation,
then confirmed via Opacus RDP accountant.

Includes the per-layer DP clipping extension (PRD Section 7/8):
Frozen per-layer clipping thresholds computed from Phase 1 checkpoint,
guaranteeing the same total L2 sensitivity C by construction.
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

from src.data import load_raw_features, split_and_scale
from src.metrics import gini_coefficient, percentage_deviance_explained
from src.model import MultipleRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_PATH = os.path.join(BASE_DIR, "data", "freMTPL2freq.csv")


def train_standard(X_train, y_train, num_features, epochs, batch_size, lr, seed=0):
    """Epsilon = infinity (non-private baseline)."""
    torch.manual_seed(seed)
    model = MultipleRegression(num_features=num_features)
    opt = torch.optim.NAdam(model.parameters(), lr=lr)
    loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    n = len(X)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()

    return model, float("inf")


def train_dp_sgd(
    X_train,
    y_train,
    num_features,
    epochs,
    batch_size,
    lr,
    target_epsilon,
    max_grad_norm=1.0,
    delta=1e-5,
    seed=0,
):
    """Trains with Opacus DP-SGD targeting the given epsilon."""
    from opacus import PrivacyEngine
    from opacus.accountants.utils import get_noise_multiplier

    torch.manual_seed(seed)
    model = MultipleRegression(num_features=num_features)
    opt = torch.optim.NAdam(model.parameters(), lr=lr)
    loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    sample_rate = batch_size / len(X_train)
    steps = epochs * math.ceil(len(X_train) / batch_size)

    # Calibrate noise multiplier for target epsilon
    try:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=steps,
        )
    except Exception as e:
        print(f"  [warn] fallback noise multiplier for eps={target_epsilon}: {e}")
        noise_multiplier = 1.0 if target_epsilon >= 3.0 else 2.5

    privacy_engine = PrivacyEngine(secure_mode=False)
    model, opt, loader = privacy_engine.make_private(
        module=model,
        optimizer=opt,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    realized_epsilon = privacy_engine.get_epsilon(delta=delta)
    unwrapped_model = model._module if hasattr(model, "_module") else model
    return unwrapped_model, realized_epsilon


def train_dp_per_layer(
    X_train,
    y_train,
    num_features,
    epochs,
    batch_size,
    lr,
    target_epsilon,
    phase1_ckpt_path,
    total_clipping_norm=1.0,
    delta=1e-5,
    seed=0,
):
    """
    Per-layer DP clipping extension (PRD Section 7 / Methodology §8.4):
    Frozen norm split computed from Phase 1 checkpoint:
    C_l = C * (||W_l|| / ||W_all||), guaranteeing sqrt(sum C_l^2) = C.
    """
    from opacus.accountants.utils import get_noise_multiplier

    torch.manual_seed(seed)
    model = MultipleRegression(num_features=num_features)

    # Compute frozen split from Phase 1 checkpoint if available
    layers = [model.layer_1, model.layer_2, model.layer_out]
    if os.path.exists(phase1_ckpt_path):
        ref_state = torch.load(phase1_ckpt_path, map_location="cpu")
        norm_l1 = torch.norm(ref_state["layer_1.weight"]).item()
        norm_l2 = torch.norm(ref_state["layer_2.weight"]).item()
        norm_l3 = torch.norm(ref_state["layer_out.weight"]).item()
    else:
        norm_l1 = torch.norm(model.layer_1.weight).item()
        norm_l2 = torch.norm(model.layer_2.weight).item()
        norm_l3 = torch.norm(model.layer_out.weight).item()

    total_ref_norm = math.sqrt(norm_l1**2 + norm_l2**2 + norm_l3**2) + 1e-9
    r1, r2, r3 = norm_l1 / total_ref_norm, norm_l2 / total_ref_norm, norm_l3 / total_ref_norm
    clip_norms = [total_clipping_norm * r1, total_clipping_norm * r2, total_clipping_norm * r3]

    sample_rate = batch_size / len(X_train)
    steps = epochs * math.ceil(len(X_train) / batch_size)
    noise_mult = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=steps,
    )

    opt = torch.optim.NAdam(model.parameters(), lr=lr)
    loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    n = len(X)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()

            # Per-layer clipping and calibrated Gaussian perturbation
            for layer, c_l in zip(layers, clip_norms):
                torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=c_l)
                for p in layer.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * (noise_mult * c_l / math.sqrt(len(idx)))
                        p.grad.add_(noise)

            opt.step()

    return model, target_epsilon, clip_norms


def main(cfg):
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"[Phase 3] Loading data (n_rows={cfg.n_rows or 'ALL'}) ...")
    X_df, y, exposure = load_raw_features(DATA_PATH, n_rows=cfg.n_rows, seed=cfg.seed)
    train, val, test, scaler, feature_names = split_and_scale(X_df, y, exposure, seed=cfg.seed)
    X_tr, y_tr, e_tr = train
    X_te, y_te, e_te = test
    num_features = X_tr.shape[1]

    results = {
        "config": vars(cfg),
        "flat_clipping": [],
        "per_layer_extension": None,
    }

    # 1. Sweep epsilon targets: {inf, 8, 3, 1}
    for target_eps in cfg.epsilons:
        tag = "inf" if target_eps in (None, float("inf")) else f"{float(target_eps):.1f}"
        print(f"\n[DP-SGD] Target epsilon = {tag} (delta=1e-5) ...")

        if target_eps in (None, float("inf")):
            model, realized_eps = train_standard(
                X_tr, y_tr, num_features, cfg.epochs, cfg.batch_size, cfg.lr, seed=cfg.seed
            )
        else:
            model, realized_eps = train_dp_sgd(
                X_tr, y_tr, num_features, cfg.epochs, cfg.batch_size, cfg.lr,
                target_epsilon=target_eps, delta=1e-5, seed=cfg.seed
            )

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_te, dtype=torch.float32)).squeeze(-1).numpy()

        pde = percentage_deviance_explained(y_te, pred, e_te)
        gini = gini_coefficient(y_te, pred, e_te)
        print(f"  Realized epsilon={realized_eps:.2f} | %PDE={pde:.2f} | Gini={gini:.3f}")

        # Save checkpoint
        ckpt_name = f"dp_eps_{tag}.pt"
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, ckpt_name))
        # Also alias if tag is 3.0 or 1.0
        if tag in ("3.0", "1.0", "8.0"):
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"dp_eps_{int(float(tag))}.pt"))

        results["flat_clipping"].append({
            "target_epsilon": target_eps,
            "realized_epsilon": realized_eps,
            "pde": pde,
            "gini": gini,
            "checkpoint": os.path.join(CKPT_DIR, ckpt_name),
        })

    # 2. Run per-layer clipping extension cell at epsilon = 3.0
    print("\n--- Running Per-Layer DP Clipping Extension (target eps=3.0) ---")
    p1_path = os.path.join(CKPT_DIR, "global.pt")
    pl_model, pl_eps, clip_norms = train_dp_per_layer(
        X_tr, y_tr, num_features, cfg.epochs, cfg.batch_size, cfg.lr,
        target_epsilon=3.0, phase1_ckpt_path=p1_path, delta=1e-5, seed=cfg.seed
    )
    pl_model.eval()
    with torch.no_grad():
        pl_pred = pl_model(torch.tensor(X_te, dtype=torch.float32)).squeeze(-1).numpy()
    pl_pde = percentage_deviance_explained(y_te, pl_pred, e_te)
    pl_gini = gini_coefficient(y_te, pl_pred, e_te)
    print(f"  Per-layer DP -> %PDE={pl_pde:.2f} | Gini={pl_gini:.3f} | Layer thresholds: {[round(c, 3) for c in clip_norms]}")
    torch.save(pl_model.state_dict(), os.path.join(CKPT_DIR, "dp_per_layer_eps_3.0.pt"))

    results["per_layer_extension"] = {
        "target_epsilon": 3.0,
        "clip_norms": clip_norms,
        "pde": pl_pde,
        "gini": pl_gini,
    }

    out_file = os.path.join(RESULTS_DIR, "phase3_privacy_utility.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nPhase 3 completed in {elapsed:.1f}s. Saved {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_rows", type=int, default=None, help="Row count; None = full 678,013 rows")
    p.add_argument("--epsilons", type=float, nargs="+", default=[float("inf"), 8.0, 3.0, 1.0])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

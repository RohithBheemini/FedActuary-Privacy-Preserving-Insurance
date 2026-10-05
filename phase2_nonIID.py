"""
Phase 2 (PRD Section 7/8): non-IID heterogeneity x FedAvg-vs-FedProx.

Evaluates two split strategies locked in PRD Section 7:
  1. Label-shift: Dirichlet over ClaimNb, alpha in {0.1, 0.5, 1, 5}
  2. Feature-shift: Dirichlet over Region, alpha = 0.5 (single value)

Compares FedAvg (mu=0.0) with FedProx (mu=0.01) across splits.
Index alignment bug is resolved by carrying metadata via split_indices.
Saves non-IID checkpoints for consumption by Phase 4 and Phase 5.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from src.data import load_raw_features, split_and_scale
from src.partition import dirichlet_partition_by_label, partition_summary
from src.train import Client, run_federated

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_PATH = os.path.join(BASE_DIR, "data", "freMTPL2freq.csv")


def build_clients(X, y, exposure, partition_labels, num_insurers, alpha, lr, batch_size, seed):
    parts = dirichlet_partition_by_label(partition_labels, num_insurers, alpha, seed=seed)
    clients = [
        Client(X[idx], y[idx], X.shape[1], lr=lr, batch_size=min(batch_size, len(idx)), seed=seed + i)
        for i, idx in enumerate(parts)
    ]
    summary = partition_summary(parts, partition_labels)
    return clients, summary, parts


def main(cfg):
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"[Phase 2] Loading data (n_rows={cfg.n_rows or 'ALL'}) ...")
    X_df, y, exposure, metadata = load_raw_features(
        DATA_PATH, n_rows=cfg.n_rows, seed=cfg.seed, return_metadata=True
    )
    train, val, test, scaler, feature_names, indices = split_and_scale(
        X_df, y, exposure, seed=cfg.seed, return_indices=True
    )
    X_tr, y_tr, e_tr = train
    X_te, y_te, e_te = test
    idx_tr = indices["train"]

    # Index alignment guaranteed:
    # y_tr is already y[idx_tr]; ClaimNb categories are int [0..4]
    claimnb_tr = y_tr.astype(int)
    # Region categories mapped precisely using train row indices
    region_tr = metadata["Region"][idx_tr]

    print(f"[Phase 2] train={len(X_tr)} test={len(X_te)} features={X_tr.shape[1]}")

    results = {
        "config": vars(cfg),
        "label_shift_claimnb": [],
        "feature_shift_region": [],
    }

    # 1. Sweep ClaimNb label-shift: alpha in {0.1, 0.5, 1, 5} x {FedAvg, FedProx}
    print("\n--- Running ClaimNb Label-Shift Sweep ---")
    for alpha in cfg.alphas:
        for mu in cfg.mus:
            method_name = "FedAvg" if mu == 0.0 else f"FedProx(mu={mu})"
            print(f"\n[ClaimNb Split] alpha={alpha} | method={method_name}")
            clients, summary, parts = build_clients(
                X_tr, y_tr, e_tr, claimnb_tr, cfg.num_insurers, alpha, cfg.lr, cfg.batch_size, cfg.seed
            )
            sizes = [s["n"] for s in summary]
            print(f"  Client sample sizes: min={min(sizes)}, max={max(sizes)}, total={sum(sizes)}")

            model, history = run_federated(
                clients, cfg.rounds, cfg.epochs, test, log_every=cfg.log_every, mu=mu
            )
            final = history[-1] if history else {"pde": float("nan"), "gini": float("nan")}
            print(f"  Result -> %PDE={final['pde']:.2f} | Gini={final['gini']:.3f}")

            res_entry = {
                "split_type": "label_shift_claimnb",
                "alpha": alpha,
                "mu": mu,
                "method": method_name,
                "client_sizes": sizes,
                "final_pde": final["pde"],
                "final_gini": final["gini"],
                "history": history,
            }
            results["label_shift_claimnb"].append(res_entry)

            # Save canonical non-IID checkpoint at alpha=0.5 (used by Phase 4 & Phase 5)
            if alpha == 0.5 and mu == 0.0:
                ckpt_path = os.path.join(CKPT_DIR, "non_iid_alpha_0.5.pt")
                torch.save(model.state_dict(), ckpt_path)
                # Also save alias
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, "non_iid_claimnb_0.5.pt"))
                print(f"  Saved checkpoint: {ckpt_path}")

    # 2. Sweep Region feature-shift: alpha = 0.5 x {FedAvg, FedProx}
    print("\n--- Running Region Feature-Shift (alpha=0.5) ---")
    for mu in cfg.mus:
        method_name = "FedAvg" if mu == 0.0 else f"FedProx(mu={mu})"
        print(f"\n[Region Split] alpha=0.5 | method={method_name}")
        clients, summary, parts = build_clients(
            X_tr, y_tr, e_tr, region_tr, cfg.num_insurers, 0.5, cfg.lr, cfg.batch_size, cfg.seed
        )
        sizes = [s["n"] for s in summary]
        print(f"  Client sample sizes: min={min(sizes)}, max={max(sizes)}, total={sum(sizes)}")

        model, history = run_federated(
            clients, cfg.rounds, cfg.epochs, test, log_every=cfg.log_every, mu=mu
        )
        final = history[-1] if history else {"pde": float("nan"), "gini": float("nan")}
        print(f"  Result -> %PDE={final['pde']:.2f} | Gini={final['gini']:.3f}")

        res_entry = {
            "split_type": "feature_shift_region",
            "alpha": 0.5,
            "mu": mu,
            "method": method_name,
            "client_sizes": sizes,
            "final_pde": final["pde"],
            "final_gini": final["gini"],
            "history": history,
        }
        results["feature_shift_region"].append(res_entry)

    out_file = os.path.join(RESULTS_DIR, "phase2_heterogeneity.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nPhase 2 completed in {elapsed:.1f}s. Saved {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_rows", type=int, default=None, help="Row count; None = full 678,013 rows")
    p.add_argument("--num_insurers", type=int, default=10)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0, 5.0])
    p.add_argument("--mus", type=float, nargs="+", default=[0.0, 0.01])
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

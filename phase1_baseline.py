"""
Phase 1 baseline (PRD Section 8, Objective 1): Global / Partial / Federated.

Config-driven per PRD Section 12 item 4 -- one entry point, dev-scale and
full-scale are just different arguments, not different scripts. This run
is DEV SCALE (small data subset, few rounds) -- it exists to prove the
pipeline produces a sane, real %PDE against real data before spending the
~1-2 hours the full 350-round x 10-insurer run needs (PRD Section 7/12
item 1). Persists checkpoints + results per PRD Section 8/12 item 3.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from src.data import (load_raw_features, save_preprocessing_artifacts,
                       split_and_scale, split_for_clients)
from src.model import MultipleRegression
from src.train import Client, evaluate, run_federated, safe_load_torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_PATH = os.path.join(BASE_DIR, "data", "freMTPL2freq.csv")


def main(cfg):
    torch.manual_seed(cfg.seed)
    t0 = time.time()

    print(f"[data] loading (n_rows={cfg.n_rows or 'ALL'}) ...")
    X_df, y, exposure = load_raw_features(DATA_PATH, n_rows=cfg.n_rows, seed=cfg.seed)
    train, val, test, scaler, feature_names = split_and_scale(X_df, y, exposure, seed=cfg.seed)
    X_tr, y_tr, e_tr = train
    X_te, y_te, e_te = test
    num_features = X_tr.shape[1]
    print(f"[data] train={len(X_tr)} val={len(val[0])} test={len(X_te)} features={num_features}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_preprocessing_artifacts(scaler, feature_names, CKPT_DIR)

    results = {"config": vars(cfg)}

    # ---------------- Global: one model, all pooled training data ----------------
    global_epochs = cfg.rounds
    global_final_ckpt = os.path.join(CKPT_DIR, "global.pt")
    global_resume_ckpt = os.path.join(CKPT_DIR, "p1_global_resume.pt")
    g_client = Client(X_tr, y_tr, num_features, lr=cfg.lr, batch_size=cfg.batch_size, seed=cfg.seed)

    if cfg.resume and os.path.exists(global_final_ckpt) and not os.path.exists(global_resume_ckpt):
        print(f"\n[RESUME] Global baseline model already completed ({global_final_ckpt}). Loading...")
        g_client.model.load_state_dict(safe_load_torch(global_final_ckpt))
        g_metrics = evaluate(g_client.model, X_te, y_te, e_te)
        print(f"[global] test %PDE={g_metrics['pde']:.2f}  Gini={g_metrics['gini']:.3f}")
        results["global"] = g_metrics
    else:
        start_ep = 1
        if cfg.resume and os.path.exists(global_resume_ckpt):
            try:
                g_save = safe_load_torch(global_resume_ckpt)
                start_ep = g_save.get("epoch", 0) + 1
                g_client.load_state(g_save["state"])
                print(f"\n[RESUME] Resuming Global model from epoch {start_ep}/{global_epochs}...")
            except Exception as e:
                print(f"[WARN] Could not read Global resume checkpoint: {e}. Starting fresh.")

        if start_ep <= global_epochs:
            print(f"\n[global] training ({global_epochs} epochs, starting from {start_ep}) ...")
            def on_g_epoch(ep, loss):
                if ep % cfg.log_every == 0 or ep == global_epochs:
                    torch.save({"epoch": ep, "state": g_client.get_state()}, global_resume_ckpt)
                    print(f"  epoch {ep:>4}/{global_epochs}: train_loss={loss:.4f}")

            g_client.fit_epochs(global_epochs, on_epoch=on_g_epoch, start_epoch=start_ep)
            if os.path.exists(global_resume_ckpt):
                try:
                    os.remove(global_resume_ckpt)
                except Exception:
                    pass

        g_metrics = evaluate(g_client.model, X_te, y_te, e_te)
        torch.save(g_client.model.state_dict(), global_final_ckpt)
        print(f"[global] test %PDE={g_metrics['pde']:.2f}  Gini={g_metrics['gini']:.3f}")
        results["global"] = g_metrics

    # ---------------- Partial: 10 independent models, no collaboration ----------------
    partial_epochs = cfg.rounds * cfg.epochs
    print(f"\n[partial] training 10 independent insurers ({partial_epochs} epochs each) ...")
    shards = split_for_clients(X_tr, y_tr, e_tr, cfg.num_insurers, seed=cfg.seed)
    partial_pdes = []
    for i, (Xs, ys, es) in enumerate(shards):
        p_ckpt = os.path.join(CKPT_DIR, f"partial_{i}.pt")
        if cfg.resume and os.path.exists(p_ckpt):
            print(f"  [RESUME] Insurer {i} already trained ({p_ckpt}). Loading...")
            c = Client(Xs, ys, num_features)
            c.model.load_state_dict(safe_load_torch(p_ckpt))
            m = evaluate(c.model, X_te, y_te, e_te)
            partial_pdes.append(m["pde"])
            print(f"  insurer {i}: n={len(Xs):>6}  %PDE={m['pde']:.2f}  Gini={m['gini']:.3f}")
        else:
            c = Client(Xs, ys, num_features, lr=cfg.lr, batch_size=min(cfg.batch_size, len(Xs)), seed=cfg.seed + i)
            c.fit(partial_epochs)
            m = evaluate(c.model, X_te, y_te, e_te)
            partial_pdes.append(m["pde"])
            torch.save(c.model.state_dict(), p_ckpt)
            print(f"  insurer {i}: n={len(Xs):>6}  %PDE={m['pde']:.2f}  Gini={m['gini']:.3f}")

    results["partial"] = {"pde_avg": float(np.mean(partial_pdes)), "pde_best": float(np.max(partial_pdes)),
                           "pde_per_insurer": partial_pdes}
    print(f"[partial] avg %PDE={results['partial']['pde_avg']:.2f}  best={results['partial']['pde_best']:.2f}")

    # ---------------- Federated: FedAvg across the same 10 shards ----------------
    print(f"\n[federated] {cfg.rounds} rounds x {cfg.epochs} local epochs, {cfg.num_insurers} insurers ...")
    clients = [Client(Xs, ys, num_features, lr=cfg.lr, batch_size=min(cfg.batch_size, len(Xs)), seed=cfg.seed + i)
               for i, (Xs, ys, es) in enumerate(shards)]
    history = []
    def log(rec):
        history.append(rec)
        print(f"  round {rec['round']:>4}/{cfg.rounds}: train_loss={rec['train_loss']:.4f}  "
              f"test %PDE={rec['pde']:.2f}  Gini={rec['gini']:.3f}")

    fed_resume_ckpt = os.path.join(CKPT_DIR, "p1_federated_resume.pt")
    fed_model, history = run_federated(
        clients,
        cfg.rounds,
        cfg.epochs,
        test,
        log_every=cfg.log_every,
        on_round=log,
        resume_checkpoint_path=fed_resume_ckpt if cfg.resume else None,
        save_checkpoint_path=fed_resume_ckpt if cfg.resume else None
    )
    torch.save(fed_model.state_dict(), os.path.join(CKPT_DIR, "federated.pt"))
    results["federated"] = {"history": history, "final": history[-1] if history else None}

    # ---------------- summary ----------------
    with open(os.path.join(RESULTS_DIR, "phase1_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if history:
        import pandas as pd
        pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, "phase1_federated_rounds.csv"), index=False)

    elapsed = time.time() - t0
    run_desc = "Full 678K run" if cfg.n_rows is None else f"Dataset n={cfg.n_rows}"
    print(f"\n{'='*60}")
    print(f"PHASE 1 SUMMARY ({run_desc}, {elapsed/60:.1f}m)")
    print(f"{'='*60}")
    print(f"{'Scenario':<14}{'%PDE':>10}   (paper benchmark)")
    print(f"{'Global':<14}{g_metrics['pde']:>10.2f}   (5.57)")
    print(f"{'Federated':<14}{results['federated']['final']['pde']:>10.2f}   (5.34)")
    print(f"{'Partial avg':<14}{results['partial']['pde_avg']:>10.2f}   (3.20-3.82)")
    print(f"{'Partial best':<14}{results['partial']['pde_best']:>10.2f}   (3.20-3.82)")
    print(f"\nsaved: {CKPT_DIR}/*.pt, {RESULTS_DIR}/phase1_metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_rows", type=int, default=None, help="Row count; None = full 678,013 rows")
    p.add_argument("--num_insurers", type=int, default=10)
    p.add_argument("--rounds", type=int, default=350)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=True, help="Automatically resume interrupted training")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Force starting fresh from round 1")
    main(p.parse_args())


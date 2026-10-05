"""
Training core shared by all scenarios (PRD Section 8).

Matches the base study's verified training setup: NAdam, lr=0.001,
batch size 500 with shuffling, PoissonNLLLoss(log_input=False, full=True)
comparing the model output DIRECTLY against y = ClaimNb (PRD Section 5).

Federated learning here is an in-process FedAvg/FedProx simulation:
sample-weighted averaging of client weights each round, with each client
keeping its own persistent NAdam state across rounds (as the base study's
client does). FedProx adds client-side proximal regularization mu/2 * ||w - w_t||^2.
"""
import copy
import os

import numpy as np
import torch
from torch import nn

from .metrics import gini_coefficient, percentage_deviance_explained
from .model import MultipleRegression

torch.set_num_threads(4)


class Client:
    """One insurer (or the Global pooled 'insurer'): owns its data, its model,
    and a persistent optimizer."""

    def __init__(self, X, y, num_features, lr=0.001, batch_size=500, seed=0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.n = len(self.X)
        self.batch_size = max(1, min(batch_size, self.n))
        self.model = MultipleRegression(num_features=num_features)
        self.opt = torch.optim.NAdam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)
        self.gen = torch.Generator().manual_seed(seed)

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_state(self):
        """Returns deep copies of model weights and optimizer state."""
        return {
            "model": copy.deepcopy(self.model.state_dict()),
            "opt": copy.deepcopy(self.opt.state_dict())
        }

    def load_state(self, state):
        """Restores model weights and optimizer state."""
        if "model" in state:
            self.model.load_state_dict(state["model"])
        if "opt" in state and state["opt"] is not None:
            self.opt.load_state_dict(state["opt"])

    def fit(self, epochs, mu=0.0):
        """Standard epoch fitting without callback."""
        return self.fit_epochs(epochs, mu=mu)

    def fit_epochs(self, epochs, mu=0.0, on_epoch=None, start_epoch=1):
        """mu > 0 adds FedProx's proximal term, client-side only -- per PRD
        Section 7 / Methodology Section 7.2. mu=0 is standard FedAvg.
        Supports resume via start_epoch and periodic callback via on_epoch."""
        self.model.train()
        global_params = [p.detach().clone() for p in self.model.parameters()] if mu > 0 else None
        last = float("nan")

        for ep in range(start_epoch, epochs + 1):
            perm = torch.randperm(self.n, generator=self.gen)
            for i in range(0, self.n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                self.opt.zero_grad()
                pred = self.model(self.X[idx])
                loss = self.loss_fn(pred, self.y[idx])

                if mu > 0 and global_params is not None:
                    prox = sum((p - gp).pow(2).sum() for p, gp in zip(self.model.parameters(), global_params))
                    loss = loss + (mu / 2.0) * prox

                loss.backward()
                # Gradient clipping for numerical stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.opt.step()
                last = loss.item()

            if on_epoch:
                on_epoch(ep, last)

        return last


def predict(model, X):
    """Generates count predictions from MultipleRegression model."""
    model.eval()
    with torch.no_grad():
        x_t = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        pred = model(x_t).squeeze(-1).detach().cpu().numpy()
        return np.atleast_1d(pred)


def evaluate(model, X, y, exposure):
    """Computes %PDE and Actuarial Gini coefficient on test split."""
    pred = predict(model, X)
    return {
        "pde": percentage_deviance_explained(y, pred, exposure),
        "gini": gini_coefficient(y, pred, exposure),
    }


def fedavg(states, weights):
    """Sample-weighted federated averaging of client state_dicts."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    avg = copy.deepcopy(states[0])
    for k in avg:
        avg[k] = sum(float(wi) * s[k].float() for wi, s in zip(w, states))
    return avg


def safe_load_torch(path):
    """Safely loads torch checkpoint, handling PyTorch 2.6+ weights_only defaults."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_federated(
    clients,
    rounds,
    local_epochs,
    test,
    log_every=1,
    on_round=None,
    mu=0.0,
    resume_checkpoint_path=None,
    save_checkpoint_path=None
):
    """FedAvg / FedProx over `clients` for `rounds` rounds.
    Supports seamless pausing and resuming across machine shutdowns.
    Returns the final global model and per-round metric history."""
    X_te, y_te, e_te = test
    global_model = MultipleRegression(num_features=clients[0].X.shape[1])
    global_state = copy.deepcopy(global_model.state_dict())
    history = []
    start_round = 1

    # Check for existing checkpoint to resume from
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        try:
            ckpt = safe_load_torch(resume_checkpoint_path)
            saved_round = ckpt.get("round", 0)
            if saved_round < rounds:
                start_round = saved_round + 1
                global_state = ckpt["global_state"]
                history = ckpt.get("history", [])
                if "client_states" in ckpt:
                    for c, s in zip(clients, ckpt["client_states"]):
                        c.load_state(s)
                print(f"[RESUME] Loaded checkpoint at round {saved_round}/{rounds}. Continuing from round {start_round}...")
            elif saved_round >= rounds:
                print(f"[RESUME] Checkpoint already completed all {rounds} rounds. Loading final state.")
                global_state = ckpt["global_state"]
                history = ckpt.get("history", [])
                global_model.load_state_dict(global_state)
                return global_model, history
        except Exception as e:
            print(f"[WARN] Failed to load resume checkpoint {resume_checkpoint_path}: {e}. Starting fresh.")

    for r in range(start_round, rounds + 1):
        states, sizes, losses = [], [], []
        for c in clients:
            c.set_weights(global_state)
            loss_val = c.fit(local_epochs, mu=mu)
            losses.append(loss_val)
            states.append(copy.deepcopy(c.model.state_dict()))
            sizes.append(c.n)

        global_state = fedavg(states, sizes)

        if r % log_every == 0 or r == rounds:
            global_model.load_state_dict(global_state)
            m = evaluate(global_model, X_te, y_te, e_te)
            rec = {"round": r, "train_loss": float(np.mean(losses)), **m}
            history.append(rec)
            if save_checkpoint_path:
                ckpt_to_save = {
                    "round": r,
                    "global_state": global_state,
                    "client_states": [c.get_state() for c in clients],
                    "history": history,
                }
                torch.save(ckpt_to_save, save_checkpoint_path)

            if on_round:
                on_round(rec)

    global_model.load_state_dict(global_state)
    # Remove temporary resume file once run is fully complete
    if save_checkpoint_path and os.path.exists(save_checkpoint_path):
        try:
            os.remove(save_checkpoint_path)
        except Exception:
            pass

    return global_model, history

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

    def fit(self, epochs, mu=0.0):
        """mu > 0 adds FedProx's proximal term, client-side only -- per PRD
        Section 7 / Methodology Section 7.2. mu=0 is standard FedAvg."""
        self.model.train()
        global_params = [p.detach().clone() for p in self.model.parameters()] if mu > 0 else None
        last = float("nan")

        for _ in range(epochs):
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


def run_federated(clients, rounds, local_epochs, test, log_every=1, on_round=None, mu=0.0):
    """FedAvg / FedProx over `clients` for `rounds` rounds.
    Returns the final global model and per-round metric history."""
    X_te, y_te, e_te = test
    global_model = MultipleRegression(num_features=clients[0].X.shape[1])
    global_state = copy.deepcopy(global_model.state_dict())
    history = []

    for r in range(1, rounds + 1):
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
            if on_round:
                on_round(rec)

    global_model.load_state_dict(global_state)
    return global_model, history

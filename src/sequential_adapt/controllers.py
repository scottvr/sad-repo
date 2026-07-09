"""Model B (coefficient controller) and its deterministic fallback.

The controller maps a context embedding (frozen-base representation of "which
domain are we in") to a flat coefficient vector over the shared adapter bases.
Gating (Model C) lives in the coefficient fitting itself (per-site gate
logits, see AdapterBank); the controller predicts the effective, already-gated
coefficients.

With only a handful of tasks the MLP necessarily memorizes; that is the
smallest truthful version. The interesting question it still answers: can a
smooth function of frozen-base context embeddings separate the tasks'
coefficient vectors at all, including on held-out context phrasings.
"""

import torch
import torch.nn as nn


class LookupController:
    """Deterministic fallback: exact task name -> fitted coefficients."""

    def __init__(self):
        self.table = {}

    def add(self, task_name, flat_coeffs):
        self.table[task_name] = flat_coeffs.detach().clone()

    def predict(self, task_name=None, context_emb=None):
        if task_name not in self.table:
            raise KeyError(f"Unknown task {task_name!r}")
        return self.table[task_name]


class MLPController(nn.Module):
    """context embedding [d_model] -> flat coefficients [n_sites * K]."""

    def __init__(self, d_model, coeff_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, coeff_dim),
        )

    def forward(self, emb):
        return self.net(emb)

    def predict(self, task_name=None, context_emb=None):
        with torch.no_grad():
            return self.forward(context_emb)


def train_mlp_controller(controller, contexts, targets, steps=500, lr=1e-2):
    """Regression: contexts [N, d_model] -> targets [N, coeff_dim] (MSE).

    Returns list of losses (for sanity checking convergence).
    """
    opt = torch.optim.Adam(controller.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        pred = controller(contexts)
        loss = nn.functional.mse_loss(pred, targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

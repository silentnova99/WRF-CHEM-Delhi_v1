"""T-GCN (Zhao et al. 2019) spatial-temporal network + hybrid loss.

Architecture
------------
A GCN(adj) block performs spatial message passing per time-step; a GRU-like
gate cell blends the message-passed input with the recurrent state. The
final hidden state maps to per-node output (pollutant). Loss is hybrid:

    L = WRMSE(y, ŷ) + λ_spe · SPE(y, ŷ)

where SPE penalizes (1 − spatial Pearson correlation per time-step) so the
model is prevented from shrinking toward a purely station-local answer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TemporalGCN(nn.Module):
    """T = time window, N = nodes, F = input features."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 24,
        out_dim: int = 1,
        dropout: float = 0.1,
        adj: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.drop = nn.Dropout(dropout)
        self.act = nn.Tanh()

        # input-to-hidden graph conv: A X W_in
        self.w_in = nn.Parameter(torch.zeros((n_features, hidden)))
        self.b_in = nn.Parameter(torch.zeros(hidden))
        # hidden-to-hidden graph conv (recurrent)
        self.w_h = nn.Parameter(torch.zeros((hidden, hidden)))
        self.b_h = nn.Parameter(torch.zeros(hidden))
        # output head on final state
        self.fc = nn.Linear(hidden, out_dim)

        self._reset_memories()
        if adj is not None:
            self.register_adjacency(adj)

    def _reset_memories(self) -> None:
        nn.init.xavier_uniform_(self.w_in)
        nn.init.zeros_(self.w_h)
        nn.init.zeros_(self.b_in)
        nn.init.zeros_(self.b_h)

    def register_adjacency(self, adj: np.ndarray) -> None:
        self.register_buffer("a", torch.from_numpy(adj.astype(np.float32)))

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """x: [T, N, F] → [N, out_dim] (forecast at final context step)."""
        a = adj if adj is not None else self.a
        t, n, f = x.shape
        h = torch.zeros((n, self.hidden), device=x.device)
        for i in range(t):
            gx = torch.mm(a, x[i]) @ self.w_in.abs() + self.b_in
            gh = torch.mm(a, h) @ self.w_h.t().abs() + self.b_h
            h = self.act(self.drop(0.5 * gx + 0.5 * gh))
        return self.fc(h)


class SpatialPatternLoss(nn.Module):
    """SPE = mean over time-steps of (1 − Pearson corr across nodes)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = target.shape[0]
        p = pred.reshape(t, -1)
        y = target.reshape(t, -1)
        p = p - p.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        denom = torch.sqrt(p.pow(2).sum(1) * y.pow(2).sum(1)).clamp_min(1e-8)
        r = (p * y).sum(1) / denom
        return (1.0 - r).mean()


def hybrid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    spe_lambda: float = 0.05,
) -> torch.Tensor:
    """WRMSE + λ·SPE. pred/target: [T, N]; weights: [N] or [T, N]."""
    w = weights.reshape(1, -1) if weights.dim() == 1 else weights
    w = w / w.sum()
    wrmse = torch.sqrt(((w * (pred - target) ** 2)).sum())
    spe = SpatialPatternLoss()(pred, target)
    return wrmse + spe_lambda * spe


def next_step_predict(
    model: nn.Module, x: np.ndarray, adj: np.ndarray, device=None
) -> np.ndarray:
    """One-step-ahead prediction: evaluate GNN on a [T, N, F] window."""
    model.eval()
    dev = device or next(model.parameters()).device
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(dev)  # [1,T,N,F]
    with torch.no_grad():
        out = model(xt[0], torch.from_numpy(adj.astype(np.float32)).to(dev))  # [N,1]
    return out.cpu().numpy().ravel()
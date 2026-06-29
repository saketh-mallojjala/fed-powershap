"""FedCE-style contribution estimation for aggregation weights.

FedCE (Jiang et al., "Fair Federated Learning via Client Contribution
Estimation", CVPR 2023) estimates each client's contribution in both gradient
space and sample (data) space and turns the estimate into an aggregation weight.

This is a *practical* gradient-space implementation analogous to how
``shapley.compute_cssv`` approximates ShapFed: we measure a client's
gradient-space contribution by the alignment of its **full-model** update with
the aggregate update (cf. ShapFed's CSSV which uses only the per-class classifier
rows), and use dataset size as the sample-space proxy. Documented as a
simplification so the comparison stays honest.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


def _flat_delta(
    global_state: Dict[str, torch.Tensor],
    client_state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    parts = []
    for k, g in global_state.items():
        if not torch.is_floating_point(g):
            continue
        parts.append((client_state[k].float() - g.float()).reshape(-1))
    return torch.cat(parts) if parts else torch.zeros(1)


def fedce_weights(
    global_state: Dict[str, torch.Tensor],
    client_states: Sequence[Dict[str, torch.Tensor]],
    client_sizes: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return length-K aggregation weights from FedCE-style contributions.

    contribution_k = ((cos(Δ_k, Δ_agg) + 1) / 2)      # gradient-space alignment
    weight_k       ∝ contribution_k * |D_k|           # × sample-space proxy
    """
    K = len(client_states)
    if K == 0:
        return np.zeros(0)
    size_w = client_sizes / client_sizes.sum()

    deltas = torch.stack([_flat_delta(global_state, cs) for cs in client_states], 0)  # (K, P)
    agg = deltas.mean(dim=0)                                                          # (P,)
    num = (deltas * agg.unsqueeze(0)).sum(dim=-1)
    denom = deltas.norm(dim=-1) * agg.norm() + eps
    cos = (num / denom).cpu().numpy()
    contrib = (cos + 1.0) / 2.0                              # map [-1,1] -> [0,1]

    w = contrib * size_w
    if w.sum() < eps:
        return size_w
    return w / w.sum()

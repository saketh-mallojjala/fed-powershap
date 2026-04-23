"""Class-Specific Shapley Values (CSSV) from ShapFed (Tastan et al., IJCAI 2024).

Key idea: Shapley-style client contributions are expensive to compute over all
2^K coalitions. ShapFed sidesteps this by using the last classifier layer's
gradient — which decomposes per class — as a signal of each client's
contribution to each class. The contribution is measured by the cosine
similarity between a client's class-c gradient and the aggregate class-c
gradient. Summing across classes (optionally sharpened or clamped) gives one
scalar weight per client.

This keeps ShapFed's semantics while staying O(K * C) per round.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


def _classifier_weight_delta(
    global_state: Dict[str, torch.Tensor],
    client_state: Dict[str, torch.Tensor],
    classifier_weight_key: str,
) -> torch.Tensor:
    """Return (client_weight - global_weight) for the classifier layer.

    Shape is (num_classes, feature_dim). Row c is the "class-c gradient" used
    by CSSV. Sign is flipped from a true gradient (Δw = -η·g) but that cancels
    in cosine similarity, so we keep the update-direction convention.
    """
    g = global_state[classifier_weight_key].detach().float()
    c = client_state[classifier_weight_key].detach().float()
    return c - g


def compute_cssv(
    global_state: Dict[str, torch.Tensor],
    client_states: Sequence[Dict[str, torch.Tensor]],
    classifier_weight_key: str,
    num_classes: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute a (num_selected, num_classes) matrix of class-specific
    Shapley-like contributions.

    For client k and class c:
        φ_{k,c} = cos_sim(Δw_k^{(c)}, Δw_agg^{(c)})
    where Δw_agg = mean over selected clients of their classifier deltas.

    Returning per-class values lets callers downstream apply class weights
    (e.g. inversely proportional to class frequency) if they want.
    """
    if len(client_states) == 0:
        return np.zeros((0, num_classes))

    deltas = [
        _classifier_weight_delta(global_state, cs, classifier_weight_key)
        for cs in client_states
    ]
    stacked = torch.stack(deltas, dim=0)          # (K, C, F)
    agg = stacked.mean(dim=0)                      # (C, F)

    # Cosine sim along feature dim, independently per class.
    num = (stacked * agg.unsqueeze(0)).sum(dim=-1)            # (K, C)
    denom = stacked.norm(dim=-1) * agg.norm(dim=-1).unsqueeze(0) + eps
    cssv = (num / denom).cpu().numpy()
    return cssv


def cssv_to_weights(
    cssv: np.ndarray,
    client_sizes: np.ndarray,
    temperature: float = 1.0,
    clamp_negative: bool = True,
    ema_prev: np.ndarray | None = None,
    ema_momentum: float = 0.0,
    size_prior: bool = True,
) -> np.ndarray:
    """Reduce (K, C) CSSV matrix to a length-K aggregation weight vector.

    Steps (all optional via flags):
      1. Sum per-class contributions → per-client score.
      2. Optionally EMA-smooth with previous round's score.
      3. Clamp negatives to 0.
      4. Temperature-scaled softmax or linear normalization.
      5. Blend with dataset-size prior (standard FedAvg weighting) to avoid
         starving large clients that happen to have off-aggregate gradients
         early in training.
    """
    per_client = cssv.sum(axis=1)                              # (K,)

    if ema_prev is not None and ema_momentum > 0.0:
        per_client = ema_momentum * ema_prev + (1 - ema_momentum) * per_client

    if clamp_negative:
        per_client = np.clip(per_client, a_min=0.0, a_max=None)

    if per_client.sum() < 1e-8:
        shap_w = client_sizes / client_sizes.sum()
    elif temperature != 1.0:
        z = per_client / max(temperature, 1e-6)
        z = z - z.max()
        exp = np.exp(z)
        shap_w = exp / exp.sum()
    else:
        shap_w = per_client / per_client.sum()

    if size_prior:
        size_w = client_sizes / client_sizes.sum()
        # Geometric blend: w ∝ shap_w * size_w (so a client with zero data
        # gets zero weight regardless of its CSSV; symmetric for shap==0).
        w = shap_w * size_w
        if w.sum() < 1e-12:
            w = size_w
        w = w / w.sum()
        return w

    return shap_w

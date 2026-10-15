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


def _consensus_delta(stacked: torch.Tensor, reference: str, trim_frac: float) -> torch.Tensor:
    """Reduce the (K, C, F) stack of client classifier deltas to a single
    (C, F) consensus direction each client is then scored against.

    "mean"    - original ShapFed (sensitive to corrupted clients).
    "median"  - coordinate-wise median (robust; a few noisy clients can't move it).
    "trimmed" - drop the top/bottom ``trim_frac`` of values per coordinate, mean
                the rest (robust, smoother than median for larger K).
    """
    if reference == "median":
        return stacked.median(dim=0).values
    if reference == "trimmed":
        K = stacked.shape[0]
        k = int(K * float(trim_frac))
        if 2 * k >= K:                              # too few clients to trim → median
            return stacked.median(dim=0).values
        sorted_s, _ = torch.sort(stacked, dim=0)    # sort clients per coordinate
        return sorted_s[k:K - k].mean(dim=0)
    return stacked.mean(dim=0)                       # "mean" (default)


def compute_cssv(
    global_state: Dict[str, torch.Tensor],
    client_states: Sequence[Dict[str, torch.Tensor]],
    classifier_weight_key: str,
    num_classes: int,
    reference: str = "mean",
    trim_frac: float = 0.2,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute a (num_selected, num_classes) matrix of class-specific
    Shapley-like contributions.

    For client k and class c:
        φ_{k,c} = cos_sim(Δw_k^{(c)}, Δw_ref^{(c)})
    where Δw_ref is the consensus direction over selected clients — the mean
    (original ShapFed) or a robust median / trimmed mean (``reference``), which
    is what lets the signal down-weight label-noise clients.

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
    agg = _consensus_delta(stacked, reference, trim_frac)   # (C, F)

    # Cosine sim along feature dim, independently per class.
    num = (stacked * agg.unsqueeze(0)).sum(dim=-1)            # (K, C)
    denom = stacked.norm(dim=-1) * agg.norm(dim=-1).unsqueeze(0) + eps
    cssv = (num / denom).cpu().numpy()
    return cssv


def _cap_weights(w: np.ndarray, max_weight: float) -> np.ndarray:
    """Project ``w`` onto {x : x >= 0, sum x = 1, max x <= max_weight}, keeping
    the uncapped entries proportional to their original values.

    EXACT, not iterative. A reviewer of the submitted manuscript pointed out
    that clipping once and renormalising -- the paper's Eqs. (9)-(10) --
    does not guarantee the cap, because renormalising divides by a sum below 1
    and inflates every surviving weight. That is correct: on
    w=[.50,.30,.10,.05,.05] with cap .30 it returns a maximum of 0.375.

    The previous implementation here tried to fix that by water-filling, but
    looped a fixed ``len(w)`` times. Proportional redistribution converges
    geometrically rather than in k steps, so it also returned over-cap weights:
    on 200k random feasible cases it violated the cap 24,412 times, by up to
    +0.487.

    The exact solution needs no iteration. Sort descending; if the top j
    entries are at the cap, the rest share the remaining mass M = 1 - j*wmax in
    proportion to their original values. Scan j upward and stop at the first j
    for which the largest uncapped entry lands at or below the cap -- that j is
    the true capped set, and the result is exact.
    """
    n = len(w)
    if n == 0 or max_weight >= 1.0 or max_weight <= 0.0:
        return w
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    tot = w.sum()
    w = w / tot if tot > 0 else np.ones(n) / n
    if max_weight * n <= 1.0 + 1e-12:   # cap at or below uniform -> uniform
        return np.ones(n) / n

    order = np.argsort(-w)
    sw = w[order]
    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        rest = sw[j:]
        rsum = rest.sum()
        mass = 1.0 - j * max_weight
        if rsum <= 0:                   # everything left is zero
            out[:j] = max_weight
            out[j:] = mass / max(n - j, 1)
            break
        scaled = rest * (mass / rsum)
        if scaled[0] <= max_weight + 1e-12:
            out[:j] = max_weight
            out[j:] = scaled
            break
    else:
        out[:] = 1.0 / n

    res = np.empty(n, dtype=np.float64)
    res[order] = out
    res = np.clip(res, 0.0, max_weight)
    return res / res.sum()


def cssv_to_weights(
    cssv: np.ndarray,
    client_sizes: np.ndarray,
    temperature: float = 1.0,
    clamp_negative: bool = True,
    ema_prev: np.ndarray | None = None,
    ema_momentum: float = 0.0,
    size_prior: bool = True,
    blend_lambda: float = 1.0,
    max_weight: float = 1.0,
    unit_interval: bool = True,
) -> np.ndarray:
    """Reduce a (K, C) CSSV matrix to a length-K aggregation weight vector.

    Improved (proposed) path vs. the legacy geometric blend:
      1. Sum per-class contributions → per-client score. With ``unit_interval``
         the cosine values are first mapped to [0,1] via (cos+1)/2, so an
         off-aggregate client gets a *small* weight rather than being clamped to
         zero (this is what caused weight collapse before).
      2. Optionally EMA-smooth with the previous round's score (reputation).
      3. Temperature-scaled softmax or linear normalization → ``shap_w``.
      4. **Convex** blend with the FedAvg size prior:
             w = (1 - blend_lambda) * size_w + blend_lambda * shap_w
         ``blend_lambda = 0`` is exactly FedAvg (graceful degradation).
      5. Cap any single weight at ``max_weight`` to bound variance.
    """
    size_w = client_sizes / client_sizes.sum()

    if unit_interval:
        per_client = ((cssv + 1.0) / 2.0).sum(axis=1)          # (K,), in [0, C]
    else:
        per_client = cssv.sum(axis=1)

    if ema_prev is not None and ema_momentum > 0.0:
        per_client = ema_momentum * ema_prev + (1 - ema_momentum) * per_client

    if not unit_interval and clamp_negative:
        per_client = np.clip(per_client, a_min=0.0, a_max=None)

    if per_client.sum() < 1e-8:
        shap_w = size_w.copy()
    elif temperature != 1.0:
        z = per_client / max(temperature, 1e-6)
        z = z - z.max()
        exp = np.exp(z)
        shap_w = exp / exp.sum()
    else:
        shap_w = per_client / per_client.sum()

    if size_prior:
        lam = float(np.clip(blend_lambda, 0.0, 1.0))
        w = (1.0 - lam) * size_w + lam * shap_w               # convex blend
        w = w / w.sum()
        return _cap_weights(w, max_weight)

    return _cap_weights(shap_w, max_weight)

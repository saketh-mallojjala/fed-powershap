"""Noise-aware / quality-aware federated baselines (reviewer B, point 1).

Reviewer B: "Authors should include more baselines to compare the proposed work
against, as there are many representative work on this problem this work is
tackling. For reference: FedNoRo, FedCorr, FedA3I."

WHAT THESE ARE, STATED PLAINLY. Each function below reimplements the *weighting
and client-selection mechanism* of the cited method inside this codebase's
round structure. They are not line-by-line ports, and three things in
particular are not reproduced:

  * FedCorr's per-sample relabelling and its fine-tuning stage, which need
    full-participation passes this protocol does not run;
  * FedNoRo's stage-2 logit-adjusted distillation loss, which changes the local
    objective rather than the aggregation;
  * FedA3I's pixel-level annotation-quality model, which is defined for
    segmentation and has no direct classification analogue.

What IS reproduced is each method's decision rule for *which clients to trust
and by how much*, which is the axis this paper competes on. Every deviation is
noted at the function that makes it. Reported numbers should be described as
"mechanism-faithful reimplementation", never as the published method's score.

References
  FedNoRo  Wu et al., IJCAI 2023.  Two-stage: warm-up, then GMM on per-client
           loss to split clean/noisy, then distance-aware aggregation.
  FedCorr  Xu et al., CVPR 2022.   LID-based identification of noisy clients,
           relabelling, then ordinary FL.
  FedA3I   Liu et al., AAAI 2024.  Annotation-quality-aware aggregation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .shapley import _cap_weights


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _norm(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(w)) / max(len(w), 1)


def _flat_delta(state: Dict[str, torch.Tensor],
                global_state: Dict[str, torch.Tensor]) -> np.ndarray:
    ks = [k for k, v in global_state.items() if torch.is_floating_point(v)]
    return torch.cat([(state[k].detach().float().cpu()
                       - global_state[k].detach().float().cpu()).reshape(-1)
                      for k in ks]).numpy()


def _gmm_split(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Two-component 1-D GMM; returns P(low-mean component), i.e. P(clean).

    FedNoRo and FedCorr both separate clients by fitting a two-component
    mixture to a per-client scalar rather than thresholding it, because the
    scale of that scalar drifts over rounds. Falls back to a median split when
    the mixture will not fit (fewer than 3 points, or zero variance).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    n = len(x)
    if n < 3 or np.ptp(x) < 1e-12:
        return np.ones(n)
    try:
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(n_components=2, random_state=seed,
                            reg_covar=1e-6, n_init=2).fit(x)
        clean = int(np.argmin(g.means_.ravel()))
        return g.predict_proba(x)[:, clean]
    except Exception:
        med = float(np.median(x))
        return (x.ravel() <= med).astype(np.float64)


# ---------------------------------------------------------------------------
# FedNoRo  (Wu et al., IJCAI 2023)
# ---------------------------------------------------------------------------
def fednoro_weights(losses: np.ndarray,
                    client_states: Sequence[Dict[str, torch.Tensor]],
                    global_state: Dict[str, torch.Tensor],
                    sizes: np.ndarray,
                    round_idx: int,
                    warmup_rounds: int = 10,
                    seed: int = 0) -> np.ndarray:
    """Stage 1 warm-up on size weights; stage 2 distance-aware down-weighting.

    Faithful: the GMM split of clients into clean/noisy on per-client loss, and
    the distance-aware rule that a noisy client's weight decays with how far
    its update sits from the clean clients' mean direction.

    Not reproduced: stage 2 replaces the noisy clients' local loss with a
    logit-adjusted distillation objective. That modifies local training rather
    than aggregation, and every method here is compared under one fixed local
    solver so the aggregation rule is what differs.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    size_w = _norm(sizes)
    if round_idx < warmup_rounds:
        return size_w

    p_clean = _gmm_split(np.asarray(losses, dtype=np.float64), seed=seed)
    is_clean = p_clean >= 0.5
    if not is_clean.any() or is_clean.all():
        return size_w

    deltas = np.stack([_flat_delta(s, global_state) for s in client_states])
    centroid = deltas[is_clean].mean(axis=0)
    cn = np.linalg.norm(centroid) + 1e-12
    cos = (deltas @ centroid) / (np.linalg.norm(deltas, axis=1) * cn + 1e-12)
    # Clean clients keep their size weight; noisy ones are scaled by how well
    # they still agree with the clean consensus, floored so they never vanish.
    scale = np.where(is_clean, 1.0, np.clip((cos + 1.0) / 2.0, 0.05, 1.0))
    return _norm(size_w * scale)


# ---------------------------------------------------------------------------
# FedCorr  (Xu et al., CVPR 2022)
# ---------------------------------------------------------------------------
def _lid_score(delta: np.ndarray, reference: np.ndarray, k: int = 5) -> float:
    """Maximum-likelihood LID of one update among the others.

    FedCorr's identification signal is the local intrinsic dimensionality of a
    client's *representations*; a noisy client sits in a locally
    higher-dimensional neighbourhood. We compute the same statistic over update
    vectors, which is the information this round structure exposes without
    extra communication.
    """
    d = np.linalg.norm(reference - delta[None, :], axis=1)
    d = np.sort(d[d > 1e-12])
    if len(d) < 2:
        return 0.0
    k = min(k, len(d))
    r = d[:k]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log(r / r[-1])
    v = v[np.isfinite(v)]
    if len(v) == 0 or abs(v.sum()) < 1e-12:
        return 0.0
    return float(-len(v) / v.sum())


def fedcorr_weights(losses: np.ndarray,
                    client_states: Sequence[Dict[str, torch.Tensor]],
                    global_state: Dict[str, torch.Tensor],
                    sizes: np.ndarray,
                    round_idx: int,
                    warmup_rounds: int = 5,
                    seed: int = 0) -> np.ndarray:
    """LID-based noisy-client identification, then exclusion and reweighting.

    Faithful: the LID statistic as the identification signal, combined with
    per-client loss, and the policy of excluding identified clients from
    aggregation rather than merely down-weighting them.

    Not reproduced: the per-sample relabelling stage and the subsequent
    fine-tuning pass on the retained clients. Both require full participation
    rounds that this protocol (5 of 10 clients per round) does not run, so
    including them would change the comparison's participation setting rather
    than its aggregation rule.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    size_w = _norm(sizes)
    n = len(client_states)
    if round_idx < warmup_rounds or n < 3:
        return size_w

    deltas = np.stack([_flat_delta(s, global_state) for s in client_states])
    lid = np.array([_lid_score(deltas[i], np.delete(deltas, i, axis=0))
                    for i in range(n)], dtype=np.float64)

    # Combine the two signals on a common scale, then split with the same GMM.
    def z(a):
        a = np.asarray(a, dtype=np.float64)
        sd = a.std()
        return (a - a.mean()) / (sd + 1e-12)

    score = z(lid) + z(np.asarray(losses, dtype=np.float64))
    p_clean = _gmm_split(score, seed=seed)
    keep = p_clean >= 0.5
    if not keep.any():
        keep = score <= np.median(score)
    w = size_w * keep.astype(np.float64)
    if w.sum() <= 0:
        return size_w
    return _norm(w)


# ---------------------------------------------------------------------------
# FedA3I  (Liu et al., AAAI 2024)
# ---------------------------------------------------------------------------
def feda3i_weights(losses: np.ndarray,
                   sizes: np.ndarray,
                   round_idx: int,
                   gamma: float = 2.0,
                   max_weight: float = 1.0,
                   warmup_rounds: int = 5) -> np.ndarray:
    """Annotation-quality-aware aggregation: weight by (quality^gamma x size).

    Faithful: the central rule that a client's aggregation weight is its data
    volume scaled by an estimated annotation quality raised to a power, so
    quality differences are amplified rather than treated linearly.

    Not reproduced: FedA3I estimates quality from a pixel-level annotation
    noise model for segmentation masks. There is no classification analogue, so
    quality is estimated here from the global model's loss on the client's
    data, normalised within the round. This makes FedA3I's *signal* closer to
    the proposed method's than it is in the original paper, and the comparison
    should be read as "same signal, different weighting rule" -- which isolates
    the amplification exponent as the difference.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    size_w = _norm(sizes)
    if round_idx < warmup_rounds:
        return size_w
    L = np.asarray(losses, dtype=np.float64)
    rng = L.max() - L.min()
    q = np.ones_like(L) if rng < 1e-12 else 1.0 - (L - L.min()) / rng
    q = np.clip(q, 1e-3, 1.0) ** float(gamma)
    return _cap_weights(_norm(size_w * q), max_weight)

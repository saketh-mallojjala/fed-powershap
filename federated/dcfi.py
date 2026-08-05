"""Dual-Channel Fair Incentive & Utility-Preserving Aggregation (DCFI-UPA).

The framework separates two objectives that this project previously conflated,
and which our own measurements showed were fighting each other:

  TRAINING CHANNEL  -- optimise predictive performance. FedDyn does the
      aggregation. The reliability detector is demoted to a *safety gate*: it
      may clip or quarantine a clearly-bad update, but it never reweights
      honest clients. This is the fix for the measured result that accurate
      down-weighting COST accuracy (better detection -> worse model), because
      a label-noise client still carries usable gradient signal.

  REWARD CHANNEL    -- compensate useful, reliable and under-represented
      contributions. Runs entirely off the training path, so paying a client
      less can never degrade the model.

Design notes that follow from the earlier experiments:

* Layer-wise asymmetry. Label noise corrupts the decision boundary far more
  than the representation, so the feature extractor is aggregated normally and
  strict reliability control is applied only to the classifier layer. This
  keeps the useful part of a noisy client's update instead of discarding the
  whole client.
* Risk from three independent views. Probe loss sees data quality but is blind
  to effort; update geometry sees effort but is fooled by inflation within the
  natural spread; temporal consistency sees replay and staleness that neither
  of the others can. No single signal covers the attack surface.
* Clipping rather than down-weighting. Shrinking the suspicious *component* of
  an update preserves the rest, where a scalar weight throws all of it away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

# The adversary taxonomy the framework is evaluated against.
ATTACKS = [
    "zero_update",      # returns the global model untouched
    "stale_model",      # resends a model trained several rounds ago
    "scaled_global",    # returns c * w_global, never trains
    "random_update",    # gaussian noise of a plausible magnitude
    "copied_update",    # replays a previously-computed delta
    "low_effort",       # trains on a small fraction of its data
    "label_flip",       # trains honestly on corrupted labels
    "norm_inflation",   # trains honestly then multiplies the update by c
]

LOW, MEDIUM, HIGH = 0, 1, 2
TIER_NAMES = {LOW: "low", MEDIUM: "medium", HIGH: "high"}


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score with a floored scale.

    Median/MAD is used because the scored population may be ~40% adversarial,
    which would wreck a mean/std standardisation. But on the ~5 active clients
    of one round the MAD can collapse to nearly zero whenever three of them
    happen to cluster, and the resulting z-scores explode -- measured as the
    dominant source of false HIGH flags on honest clients (15 of 22 in a
    fully-clean run). Flooring the scale relative to the median magnitude
    bounds that: a spread estimate from five points is not trustworthy enough
    to justify an unbounded score.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    floor = max(0.25 * abs(med), 1e-6)
    scale = max(scale, floor)
    return (x - med) / scale


@dataclass
class ClientHistory:
    """Per-client state the reward and risk channels both accumulate."""
    reliability: float = 1.0          # historical reliability in [0, 1]
    selections: int = 0               # times selected (selection-fairness metric)
    accepted: int = 0                 # times accepted into aggregation
    quarantined_until: int = -1       # round index the quarantine expires
    high_risk_streak: int = 0
    suspicion: int = 0                # times the soft score alone read HIGH
    prev_delta_flat: Optional[np.ndarray] = None   # for temporal consistency
    prev_norm: float = 0.0
    paid: float = 0.0                 # cumulative released reward
    held: float = 0.0                 # cumulative held-back reward
    utility_ema: float = 0.0


@dataclass
class RoundRisk:
    """Risk assessment for one round's active set."""
    ids: List[int]
    risk: np.ndarray
    tier: np.ndarray
    components: Dict[str, np.ndarray] = field(default_factory=dict)


class RiskScorer:
    """Combines three independent views into one risk score per client.

    All three are converted to robust z-scores and summed with configurable
    weights, so no view can dominate purely because of its natural scale.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.w_loss = float(getattr(cfg, "dcfi_w_loss", 1.0))
        self.w_geom = float(getattr(cfg, "dcfi_w_geometry", 1.0))
        self.w_temporal = float(getattr(cfg, "dcfi_w_temporal", 1.0))
        self.med_thresh = float(getattr(cfg, "dcfi_medium_risk", 1.0))
        self.high_thresh = float(getattr(cfg, "dcfi_high_risk", 2.5))
        # Absolute-override parameters (see score()).
        self.gross_ratio = float(getattr(cfg, "dcfi_gross_ratio", 8.0))
        # Replay is a near-exact resend, so the bar sits just below 1.0. Honest
        # self-similarity averages ~0.98, hence the default of 0.999.
        self.replay_cos = float(getattr(cfg, "dcfi_replay_cos", 0.999))

    def score(
        self,
        ids: Sequence[int],
        probe_losses: np.ndarray,
        deltas_flat: List[np.ndarray],
        sizes: np.ndarray,
        hist: Dict[int, ClientHistory],
    ) -> RoundRisk:
        ids = list(ids)
        n = len(ids)
        comp: Dict[str, np.ndarray] = {}

        # (1) Data quality. High loss on a client's own data under the clean
        # consensus indicates corrupted labels. Blind to effort by construction.
        comp["loss"] = _robust_z(probe_losses) if n > 2 else np.zeros(n)

        # (2) Update geometry. Effort density = ||delta|| / n_k removes the
        # size confound (honest ||delta|| scales ~linearly with client size).
        # Two-sided: too small means no work, too large means inflation.
        norms = np.array([float(np.linalg.norm(d)) for d in deltas_flat])
        dens = norms / np.maximum(sizes, 1.0)
        pos = dens[dens > 1e-12]
        ref = float(np.median(pos)) if len(pos) else 1.0
        dev = np.abs(np.log((dens + 1e-12) / (ref + 1e-12)))
        comp["geometry"] = _robust_z(dev) if n > 2 else np.zeros(n)
        # Absolute flags, independent of the (small, possibly contaminated) peer
        # sample. `gross_scale` fires on an update far outside the honest spread
        # -- measured log-sd of honest effort density is ~1.0, so a 4x deviation
        # is ~1.4 sd and still plausible; we only hard-flag beyond `gross_ratio`.
        comp["zero_update"] = (dens <= 1e-12).astype(float)
        with np.errstate(divide="ignore"):
            ratio = np.where(dens > 1e-12, dens / max(ref, 1e-12), 0.0)
        comp["gross_scale"] = ((ratio > self.gross_ratio)
                               | ((ratio < 1.0 / self.gross_ratio) & (dens > 1e-12))
                               ).astype(float)

        # (3) Temporal consistency. An honest client's update direction drifts
        # smoothly as the global model moves. A replayed or stale update repeats
        # itself almost exactly (cosine ~ 1 with its own previous delta) or, if
        # stale, drifts steadily away as the global model advances past it.
        # MEASURED CORRECTION: honest consecutive updates are *already* highly
        # self-similar (mean cosine ~0.98 with their own previous delta), because
        # local data is fixed and the model is converging. An earlier version
        # flagged cos>0.9 and hit 82% of honest clients. A genuine replay resends
        # the SAME vector, so the signature is cosine ~1 to floating-point
        # precision -- that is what separates it, not mere similarity.
        temporal = np.zeros(n)
        for i, cid in enumerate(ids):
            h = hist[cid]
            d = deltas_flat[i]
            nd = float(np.linalg.norm(d))
            if h.prev_delta_flat is not None and nd > 1e-12 and h.prev_norm > 1e-12:
                cos = float(np.dot(d, h.prev_delta_flat) / (nd * h.prev_norm))
                temporal[i] = max(0.0, cos - self.replay_cos) / (1.0 - self.replay_cos)
            h.prev_delta_flat, h.prev_norm = d.copy(), nd
        comp["temporal"] = temporal

        # Soft risk drives the MEDIUM tier (clip). Temporal is excluded here --
        # it is a binary replay signature, not a graded suspicion, so it acts
        # only through the absolute override below.
        risk = (self.w_loss * np.clip(comp["loss"], 0, None)
                + self.w_geom * np.clip(comp["geometry"], 0, None))

        # NOTE: historical reliability is deliberately NOT fed back into the
        # risk score. Doing so creates a death spiral -- one unlucky flag lowers
        # reliability, which raises risk, which flags again. Measured at a 62%
        # false-flag rate on honest clients, which collapsed the training
        # channel. Reliability governs quarantine and reward only.

        tier = np.where(risk >= self.high_thresh, HIGH,
                        np.where(risk >= self.med_thresh, MEDIUM, LOW))

        # Absolute overrides. The relative z-scores above are computed over only
        # ~5 active clients, of whom several may be adversarial, so the median
        # and MAD are unreliable. These three signatures are unambiguous on
        # their own and do not need a peer comparison:
        #   * an update that is (near) exactly zero -- no work was done
        #   * an update almost identical to the client's own previous one --
        #     a replay; honest consecutive updates are correlated but not this
        #     correlated, because the global model has moved underneath them
        #   * an update many times the peer scale -- inflation or scaled-global
        hard = ((comp.get("zero_update", np.zeros(n)) > 0)
                | (comp["temporal"] > 0)
                | (comp.get("gross_scale", np.zeros(n)) > 0))

        # ESCALATION RULE. Exclusion is irreversible for the round, so it needs
        # either an unambiguous signature (`hard`) or corroboration across
        # rounds. A soft z-score alone -- which on clean data is just non-IID
        # spread measured over five clients -- earns a CLIP on first sight and
        # only escalates to exclusion if the client has been high-risk before.
        # Without this, false soft-HIGHs alone collapsed clean-data seeds.
        soft_high = (tier == HIGH) & ~hard
        # `suspicion` counts CONSECUTIVE soft-high rounds and resets on any
        # clean round -- "repeatedly high-risk", not "ever suspicious". A
        # cumulative counter was tried and is strictly worse: it never forgets,
        # so an honest client that trips the soft score once is excluded on
        # every later trip, and clean-data accuracy degrades over training.
        repeat = np.array([hist[c].suspicion >= 1 for c in ids])
        for i, c in enumerate(ids):
            hist[c].suspicion = hist[c].suspicion + 1 if soft_high[i] else 0
        tier = np.where(soft_high & ~repeat, MEDIUM, tier)
        tier = np.where(hard, HIGH, tier)
        return RoundRisk(ids=ids, risk=risk, tier=tier, components=comp)


class SafetyGate:
    """Applies the tiered response to client updates, layer-wise.

    low     -> untouched, full FedDyn participation
    medium  -> the classifier component is clipped toward the peer-median norm;
               the feature extractor is left alone
    high    -> excluded this round, and quarantined if the streak persists
    """

    def __init__(self, cfg, classifier_prefix: str):
        self.cfg = cfg
        self.classifier_prefix = classifier_prefix
        self.clip_q = float(getattr(cfg, "dcfi_clip_quantile", 0.5))
        self.quarantine_after = int(getattr(cfg, "dcfi_quarantine_after", 2))
        self.quarantine_rounds = int(getattr(cfg, "dcfi_quarantine_rounds", 5))
        self.min_keep_frac = float(getattr(cfg, "dcfi_min_keep_frac", 0.6))

    def _is_classifier(self, key: str) -> bool:
        return key.startswith(self.classifier_prefix)

    def apply(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: List[Dict[str, torch.Tensor]],
        rr: RoundRisk,
        hist: Dict[int, ClientHistory],
        round_idx: int,
    ):
        """Return (kept_states, kept_ids, actions) after gating."""
        # Reference scale for clipping: the median classifier-delta norm over
        # the low-risk clients only, so a colluding high-risk group cannot
        # inflate the very reference used to rein them in.
        ref_pool = [
            self._classifier_norm(global_state, cs)
            for cs, t in zip(client_states, rr.tier) if t == LOW
        ]
        if not ref_pool:
            ref_pool = [self._classifier_norm(global_state, cs) for cs in client_states]
        ref = float(np.median(ref_pool)) if ref_pool else 0.0

        # QUORUM FLOOR. Exclusion is the only irreversible action the gate takes,
        # and it compounds: if most of an active set is dropped, FedDyn
        # aggregates over one or two clients and the global model destabilises.
        # Measured without this floor: 3 of 5 clean-data seeds collapsed to
        # single-class prediction (0.500 balanced accuracy) purely from false
        # exclusions. So we cap how many clients the gate may drop, demoting the
        # least-risky excess back to CLIPPED. A gate that can wreck the model it
        # is protecting is not a safety mechanism.
        tiers = np.array(rr.tier, dtype=int)
        n_active = len(tiers)
        min_keep = max(1, int(np.ceil(self.min_keep_frac * n_active)))
        high_idx = [i for i in range(n_active) if tiers[i] == HIGH]
        if n_active - len(high_idx) < min_keep:
            n_demote = min_keep - (n_active - len(high_idx))
            # Demote the least risky of the HIGH set first.
            for i in sorted(high_idx, key=lambda j: rr.risk[j])[:n_demote]:
                tiers[i] = MEDIUM

        kept, kept_ids, actions = [], [], []
        for cs, cid, tier in zip(client_states, rr.ids, tiers):
            h = hist[cid]
            if tier == HIGH:
                h.high_risk_streak += 1
                if h.high_risk_streak >= self.quarantine_after:
                    h.quarantined_until = round_idx + self.quarantine_rounds
                actions.append("excluded")
                continue
            if tier == MEDIUM:
                cs = self._clip_classifier(global_state, cs, ref)
                actions.append("clipped")
                h.high_risk_streak = 0
            else:
                actions.append("full")
                h.high_risk_streak = 0
            kept.append(cs)
            kept_ids.append(cid)
        return kept, kept_ids, actions

    def _classifier_norm(self, global_state, cs) -> float:
        tot = 0.0
        for k, g in global_state.items():
            if self._is_classifier(k) and torch.is_floating_point(g) and k in cs:
                tot += float(((cs[k].float() - g.float()) ** 2).sum())
        return tot ** 0.5

    def _clip_classifier(self, global_state, cs, ref: float) -> Dict[str, torch.Tensor]:
        """Shrink only the classifier component toward the reference norm.

        The feature extractor passes through untouched: a noisy client's
        representation learning is still useful, it is the decision boundary
        that is corrupted. Discarding the whole update (as scalar
        down-weighting does) throws the useful part away too.
        """
        cur = self._classifier_norm(global_state, cs)
        if ref <= 0 or cur <= ref or cur <= 0:
            return cs
        scale = ref / cur
        out = dict(cs)
        for k, g in global_state.items():
            if self._is_classifier(k) and torch.is_floating_point(g) and k in cs:
                gf = g.float()
                out[k] = (gf + scale * (cs[k].float() - gf)).to(g.dtype)
        return out


class RewardChannel:
    """Budget split into base / utility / fairness, with caps and a holdback.

    base     - flat participation payment. Underwrites individual rationality:
               an honest client's reward covers its cost even in a round where
               its marginal utility happened to be small.
    utility  - proportional to measured validation utility.
    fairness - proportional to contribution on the metrics that collapse first
               under non-IID data (minority-class recall, worst-client
               accuracy), plus an under-participation bonus.

    A per-round cap bounds any one client's share. A holdback fraction is
    retained and released only once the client's historical reliability clears
    a threshold, so a client that games one round cannot cash out immediately.
    """

    def __init__(self, cfg):
        self.budget = float(getattr(cfg, "dcfi_budget", 1.0))
        self.w_base = float(getattr(cfg, "dcfi_w_base", 0.2))
        self.w_utility = float(getattr(cfg, "dcfi_w_utility", 0.5))
        self.w_fairness = float(getattr(cfg, "dcfi_w_fairness", 0.3))
        self.cap = float(getattr(cfg, "dcfi_reward_cap", 0.4))
        self.holdback = float(getattr(cfg, "dcfi_holdback", 0.2))
        self.release_reliability = float(getattr(cfg, "dcfi_release_reliability", 0.7))

    def pay(
        self,
        accepted_ids: List[int],
        utility: np.ndarray,
        fairness: np.ndarray,
        hist: Dict[int, ClientHistory],
    ) -> Dict[int, float]:
        """Split this round's budget. Returns {client_id: released reward}."""
        if not accepted_ids:
            return {}
        n = len(accepted_ids)

        def nz(x):
            x = np.clip(np.asarray(x, float), 0.0, None)
            s = x.sum()
            return x / s if s > 1e-12 else np.ones(n) / n

        share = (self.w_base * (np.ones(n) / n)
                 + self.w_utility * nz(utility)
                 + self.w_fairness * nz(fairness))
        share = share / share.sum()
        share = self._cap(share)

        out = {}
        for cid, s in zip(accepted_ids, share):
            amount = self.budget * s
            h = hist[cid]
            hold = amount * self.holdback
            release = amount - hold
            # Release previously-held money once reliability is re-established.
            if h.reliability >= self.release_reliability and h.held > 0:
                release += h.held
                h.held = 0.0
            h.held += hold
            h.paid += release
            out[cid] = release
        return out

    def _cap(self, share: np.ndarray) -> np.ndarray:
        """Water-filling cap so no client exceeds `cap` of the round budget."""
        if self.cap >= 1.0 or len(share) * self.cap <= 1.0:
            return share
        s = share.copy()
        for _ in range(len(s)):
            over = s > self.cap + 1e-12
            if not over.any():
                break
            excess = (s[over] - self.cap).sum()
            s[over] = self.cap
            under = ~over
            if not under.any():
                break
            s[under] += excess * (s[under] / s[under].sum())
        return s / s.sum()


def select_dcfi(
    num_clients: int,
    m: int,
    hist: Dict[int, ClientHistory],
    rng: np.random.Generator,
    round_idx: int,
    under_bonus: float = 0.5,
    utility_weight: float = 0.5,
) -> List[int]:
    """Select the active set from utility, reliability and under-participation.

    The under-participation bonus is what keeps selection *fair*: a pure
    utility-greedy rule concentrates on a few strong clients, which starves the
    rest of both model influence and reward, and shows up directly in the
    selection-frequency Jain index. Quarantined clients are skipped entirely.
    """
    eligible = [c for c in range(num_clients) if hist[c].quarantined_until < round_idx]
    if len(eligible) <= m:
        return sorted(eligible)

    sel = np.array([hist[c].selections for c in eligible], dtype=float)
    # Under-participation: how far below the most-selected client each one is.
    under = (sel.max() - sel) / max(sel.max(), 1.0)
    util = np.array([hist[c].utility_ema for c in eligible], dtype=float)
    if util.max() > util.min():
        util = (util - util.min()) / (util.max() - util.min())
    else:
        util = np.zeros_like(util)
    rel = np.array([hist[c].reliability for c in eligible], dtype=float)

    score = utility_weight * util + under_bonus * under + rel
    # Sample without replacement proportional to score, rather than taking the
    # top-m: deterministic top-m would lock out mid-ranked clients permanently.
    p = np.clip(score, 1e-6, None)
    p = p / p.sum()
    idx = rng.choice(len(eligible), size=m, replace=False, p=p)
    return sorted(int(eligible[i]) for i in idx)


def update_reliability(hist: ClientHistory, tier: int, momentum: float = 0.8) -> None:
    """EMA over per-round verdicts. Asymmetric by design: trust decays fast on a
    high-risk round and rebuilds slowly, so intermittent gaming does not pay."""
    target = {LOW: 1.0, MEDIUM: 0.5, HIGH: 0.0}[int(tier)]
    m = momentum if target >= hist.reliability else momentum * 0.5
    hist.reliability = m * hist.reliability + (1 - m) * target

"""Server: drives a round for the proposed method and all baselines.

Per round: select clients -> run the configured local solver -> compute
aggregation weights (method-specific) -> aggregate -> optional server momentum
-> load into the global model. Holds the cross-round state some methods need
(SCAFFOLD control variate, FedDyn h, FedBN per-client BatchNorm, Shapley
reputation EMA, server-momentum buffer).
"""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import get_test_loader
from federated.aggregation import _BN_MARKERS, aggregate, bn_keys
from federated.client import Client
from federated.contribution import fedce_weights
from federated.dcfi import (
    HIGH, LOW, MEDIUM, TIER_NAMES, ClientHistory, RewardChannel, RiskScorer,
    SafetyGate, select_dcfi, update_reliability,
)
from federated.selection import SelectionOutcome, select_clients
from federated.shapley import _cap_weights, compute_cssv, cssv_to_weights
from models import CLASSIFIER_LAYER_NAME


CLASSIFIER_WEIGHT_KEY = f"{CLASSIFIER_LAYER_NAME}.weight"


def _flat_delta(global_state, client_state) -> np.ndarray:
    """Flattened full-model update, as a numpy vector (for risk geometry)."""
    parts = []
    for k, g in global_state.items():
        if torch.is_floating_point(g) and k in client_state:
            parts.append((client_state[k].float() - g.float()).reshape(-1))
    if not parts:
        return np.zeros(1, dtype=np.float32)
    return torch.cat(parts).cpu().numpy()


def _split_val_test(test_set, frac: float, seed: int, num_workers: int = 0):
    """Split the test pool into a server validation set and a reporting test set.

    DCFI needs held-out data to estimate each client's utility. Reusing the
    reported test set for that would let the mechanism optimise against the
    numbers we report, so the two are kept disjoint.
    """
    from torch.utils.data import Subset
    n = len(test_set)
    idx = np.random.default_rng(seed + 99).permutation(n)
    n_val = max(1, int(round(n * frac)))
    val = Subset(test_set, idx[:n_val].tolist())
    tst = Subset(test_set, idx[n_val:].tolist())
    return (get_test_loader(val, num_workers=num_workers),
            get_test_loader(tst, num_workers=num_workers))


def _is_running_stat(key: str) -> bool:
    """BN running buffers (mean/var/count) must be averaged, never momentum- or
    correction-updated: pushing running_var negative blows up BatchNorm."""
    return key.endswith(_BN_MARKERS)


def _binary_auc(labels: np.ndarray, pos_prob: np.ndarray) -> float:
    """ROC-AUC for a binary task via the Mann-Whitney rank identity.

    Ties get averaged ranks, so this matches sklearn's roc_auc_score. Returns 0.5
    when one class is absent from the test set (AUC is undefined there).
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(pos_prob, kind="mergesort")
    sorted_p = pos_prob[order]
    ranks = np.empty(len(pos_prob), dtype=np.float64)
    i = 0
    while i < len(sorted_p):                      # average ranks within tie groups
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _binary_quality_metrics(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Beyond-accuracy metrics for a binary screening task.

    Motivation: accuracy and balanced accuracy both read the model at its
    *default* 0.5 threshold, so they conflate two very different failures --
    "the model cannot separate the classes" and "the model separates them but
    its probabilities are mis-scaled". Under label noise the second dominates,
    so we report:

      ece      - expected calibration error (15 equal-width bins). How
                 trustworthy the predicted probability is, which is what
                 actually matters for clinical triage.
      brier    - mean squared probability error (proper scoring rule).
      auprc    - average precision for the positive (depressed) class; more
                 informative than AUC when the classes are imbalanced.
      f1_pos   - F1 on the positive class at the default threshold.
      bacc_opt - balanced accuracy at the BEST threshold. The gap between this
                 and `balanced_acc` isolates threshold/calibration loss from
                 genuine loss of discriminative power.
    """
    out: Dict[str, float] = {}
    n = len(y)
    if n == 0 or not (0 < y.sum() < n):
        return out

    # --- calibration ---
    bins = np.linspace(0.0, 1.0, 16)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, 14)
    ece = 0.0
    for b in range(15):
        m = idx == b
        if m.any():
            ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    out["ece"] = float(ece)
    out["brier"] = float(np.mean((p - y) ** 2))

    # --- ranking quality on the positive class (average precision) ---
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, n + 1)
    n_pos = int(y.sum())
    out["auprc"] = float((precision * ys).sum() / n_pos)

    # --- F1 on the positive class at the default threshold ---
    tp0 = float(((pred == 1) & (y == 1)).sum())
    fp0 = float(((pred == 1) & (y == 0)).sum())
    fn0 = float(((pred == 0) & (y == 1)).sum())
    denom = 2 * tp0 + fp0 + fn0
    out["f1_pos"] = float(2 * tp0 / denom) if denom > 0 else 0.0

    # --- best achievable balanced accuracy over all thresholds ---
    # Sweeping the sorted scores: at cut i the top-i items are predicted
    # positive, so TPR = tp[i]/n_pos and FPR = (i-tp[i])/n_neg.
    n_neg = n - n_pos
    i = np.arange(1, n + 1)
    tpr = tp / n_pos
    fpr = (i - tp) / n_neg
    out["bacc_opt"] = float(np.max(0.5 * (tpr + (1.0 - fpr))))
    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12) if hi > lo else np.zeros_like(x)


class Server:
    def __init__(
        self,
        cfg,
        clients: List[Client],
        test_set,
        model_builder,
    ):
        self.cfg = cfg
        self.clients = clients
        self.model_builder = model_builder
        self.global_model = model_builder().to(cfg.device)
        self.test_loader = get_test_loader(
            test_set, num_workers=getattr(cfg, "num_workers", 0)
        )
        self.rng = np.random.default_rng(cfg.seed)

        self.client_sizes = np.array([c.size for c in clients], dtype=np.float64)
        self.cssv_ema: Optional[np.ndarray] = None  # smoothed per-client Shapley score
        self.history: List[Dict] = []

        # Method-specific cross-round server state (lazy).
        self.scaffold_c: Optional[Dict[str, torch.Tensor]] = None  # global control
        self.feddyn_h: Optional[Dict[str, torch.Tensor]] = None    # FedDyn cloud term
        self.client_bn: Dict[int, Dict[str, torch.Tensor]] = {}    # FedBN local BN
        self.momentum_buf: Optional[Dict[str, torch.Tensor]] = None

        # ---- DCFI-UPA state (inert unless cfg.dcfi) ----
        self.dcfi_hist: Dict[int, ClientHistory] = {}
        self.risk_scorer = None
        self.safety_gate = None
        self.reward_channel = None
        self.val_loader = None
        if bool(getattr(cfg, "dcfi", False)):
            self.dcfi_hist = {c.id: ClientHistory() for c in clients}
            self.risk_scorer = RiskScorer(cfg)
            self.safety_gate = SafetyGate(cfg, CLASSIFIER_LAYER_NAME)
            self.reward_channel = RewardChannel(cfg)
            # The mechanism needs a validation set to estimate utility. Carve it
            # out of the test pool and keep it DISJOINT from the reported test
            # split, so utility estimation never touches reported metrics.
            self.val_loader, self.test_loader = _split_val_test(
                test_set, frac=float(getattr(cfg, "dcfi_val_frac", 0.3)),
                seed=cfg.seed, num_workers=getattr(cfg, "num_workers", 0),
            )

    # ---------- selection ----------

    def _select_dcfi_outcome(self, round_idx: int) -> SelectionOutcome:
        """Selection driven by utility, reliability and under-participation,
        skipping quarantined clients (federated/dcfi.py::select_dcfi)."""
        active = select_dcfi(
            num_clients=len(self.clients),
            m=min(self.cfg.active_size_m, len(self.clients)),
            hist=self.dcfi_hist,
            rng=self.rng,
            round_idx=round_idx,
            under_bonus=float(getattr(self.cfg, "dcfi_under_bonus", 0.5)),
            utility_weight=float(
                getattr(self.cfg, "dcfi_selection_utility_weight", 0.5)),
        )
        return SelectionOutcome(candidates=active, active=active)

    def _select(self, round_idx: int) -> SelectionOutcome:
        if bool(getattr(self.cfg, "dcfi", False)):
            return self._select_dcfi_outcome(round_idx)
        cfg = self.cfg
        N = len(self.clients)
        strat = cfg.selection_strategy

        if strat in ("random", "full"):
            return select_clients(
                strategy=strat, num_clients=N, client_sizes=self.client_sizes,
                d=cfg.candidate_size_d, m=cfg.active_size_m, rng=self.rng,
                loss_fn=None, size_weighted_candidates=cfg.size_weighted_candidates,
            )

        # Power-of-Choice family. Anneal toward random late in training.
        frac = float(getattr(cfg, "poc_anneal", 0.0)) * (
            round_idx / max(cfg.num_rounds - 1, 1)
        )
        if frac > 0.0 and self.rng.random() < frac:
            return select_clients(
                strategy="random", num_clients=N, client_sizes=self.client_sizes,
                d=cfg.candidate_size_d, m=cfg.active_size_m, rng=self.rng,
                loss_fn=None, size_weighted_candidates=cfg.size_weighted_candidates,
            )

        outcome = select_clients(
            strategy="pow_d", num_clients=N, client_sizes=self.client_sizes,
            d=cfg.candidate_size_d, m=cfg.active_size_m, rng=self.rng,
            loss_fn=lambda cid: self.clients[cid].probe_loss(self.global_model),
            size_weighted_candidates=cfg.size_weighted_candidates,
        )

        # Reputation-aware re-ranking: keep high-loss clients but down-rank those
        # with persistently low Shapley reputation (whose updates we'd discard).
        rw = float(getattr(cfg, "reputation_weight", 0.0))
        if rw > 0.0 and self.cssv_ema is not None and outcome.candidate_losses:
            cand = np.array(outcome.candidates)
            loss_n = _minmax(np.array(outcome.candidate_losses))
            rep_n = _minmax(self.cssv_ema[cand])
            score = loss_n + rw * rep_n
            order = np.argsort(score)[::-1]
            m = min(cfg.active_size_m, len(cand))
            outcome.active = cand[order[:m]].tolist()
        return outcome

    def _round_lr(self, round_idx: int) -> float:
        """Effective local LR for this round. Cosine decay curbs the late-round
        divergence that plagues constant-LR runs under non-IID / noisy clients."""
        cfg = self.cfg
        base = float(cfg.local_lr)
        if getattr(cfg, "lr_schedule", "none") != "cosine":
            return base
        import math
        t = round_idx / max(cfg.num_rounds - 1, 1)
        frac = float(getattr(cfg, "lr_min_frac", 0.1))
        return base * (frac + (1.0 - frac) * 0.5 * (1.0 + math.cos(math.pi * t)))

    # ---------- core round ----------

    def run_round(self, round_idx: int) -> Dict:
        cfg = self.cfg
        global_state = {
            k: v.detach().cpu().clone()
            for k, v in self.global_model.state_dict().items()
        }

        outcome = self._select(round_idx)

        # Local training on active clients.
        client_states: List[Dict[str, torch.Tensor]] = []
        auxes: List[Dict] = []
        for cid in outcome.active:
            sent_state = global_state
            if cfg.aggregation == "fedbn" and cid in self.client_bn:
                sent_state = dict(global_state)
                sent_state.update(self.client_bn[cid])  # inject this client's BN
            st, aux = self.clients[cid].local_train(
                sent_state, self.model_builder,
                server_control=self.scaffold_c if cfg.aggregation == "scaffold" else None,
                lr=self._round_lr(round_idx),
            )
            client_states.append(st)
            auxes.append(aux)

        active_sizes = self.client_sizes[outcome.active]
        cssv = None

        # ----- DCFI-UPA: dual-channel path -----
        if bool(getattr(cfg, "dcfi", False)):
            d = self._dcfi_round(global_state, client_states, outcome.active, round_idx)
            self.global_model.load_state_dict(
                self._apply_server_momentum(global_state, d["new_state"]))
            diag = self._diagnostics(global_state, client_states, None)
            return {
                "round": round_idx,
                "candidates": outcome.candidates,
                "candidate_losses": outcome.candidate_losses,
                "active": outcome.active,
                "weights": np.asarray(d["weights"]).tolist(),
                "cssv": None,
                "contrib_losses": getattr(self, "_last_losses", None),
                # `weights` is surfaced at the top level already; drop the raw
                # array here so the record stays JSON-serialisable.
                "dcfi": {k: v for k, v in d.items()
                         if k not in ("new_state", "weights")},
                **diag,
            }

        # ----- compute weights + new global state, per method -----
        if cfg.aggregation == "shapfed_wa":
            new_state, weights, cssv = self._agg_shapfed(
                global_state, client_states, active_sizes, outcome.active, round_idx
            )
        elif cfg.aggregation == "shapfed_dyn":
            # Proposed: contribution-aware FedDyn aggregation. The quality signal
            # is either ShapFed CSSV or (reliable under noise) the loss detector.
            if getattr(cfg, "contrib_signal", "cssv") == "loss":
                weights = self._loss_weights(active_sizes, outcome.active, round_idx)
            else:
                weights, cssv = self._shapfed_weights(
                    global_state, client_states, active_sizes, outcome.active, round_idx
                )
            new_state = self._agg_feddyn(global_state, client_states, weights=weights)
        elif cfg.aggregation == "fedce":
            weights = fedce_weights(global_state, client_states, active_sizes)
            new_state = aggregate(client_states, weights)
        elif cfg.aggregation == "scaffold":
            weights = np.ones(len(client_states)) / len(client_states)
            new_state = aggregate(client_states, weights)
            self._update_scaffold_control(auxes)
        elif cfg.aggregation == "feddyn":
            weights = np.ones(len(client_states)) / len(client_states)
            new_state = self._agg_feddyn(global_state, client_states)
        elif cfg.aggregation in ("fedavg", "fedbn"):
            weights = active_sizes / active_sizes.sum()
            new_state = aggregate(client_states, weights)
            if cfg.aggregation == "fedbn":
                self._store_client_bn(outcome.active, client_states)
        else:
            raise ValueError(f"Unknown aggregation: {cfg.aggregation}")

        new_state = self._apply_server_momentum(global_state, new_state)
        self.global_model.load_state_dict(new_state)

        # ---- diagnostics for the incentive analysis (C2) ----
        # Computed AFTER the model update and never fed back into weights, so
        # logging them cannot change any method's behaviour. They let the reward
        # analysis compare signals (loss / cosine / norm) on identical runs.
        diag = self._diagnostics(global_state, client_states, cssv)

        return {
            "round": round_idx,
            "candidates": outcome.candidates,
            "candidate_losses": outcome.candidate_losses,
            "active": outcome.active,
            "weights": np.asarray(weights).tolist(),
            "cssv": cssv.tolist() if cssv is not None else None,
            "contrib_losses": getattr(self, "_last_losses", None),
            **diag,
        }

    # ---------- DCFI-UPA ----------

    def _dcfi_round(self, global_state, client_states, active, round_idx) -> Dict:
        """One DCFI round, after local training.

        Training channel: risk -> safety gate (clip/quarantine) -> FedDyn on the
        survivors with UNIFORM weights. Reward channel: utility and fairness are
        measured on the validation split and paid separately. The two never
        exchange numbers -- reward has no path back into the model.
        """
        cfg = self.cfg
        hist = self.dcfi_hist
        for cid in active:
            hist[cid].selections += 1

        # --- risk assessment ---
        flat = [_flat_delta(global_state, cs) for cs in client_states]
        # Probe the CURRENT global model on each active client's own data. Must
        # be recomputed every round -- self.global_model still holds this
        # round's starting weights, since the new state is loaded after.
        losses = np.array(
            [self.clients[c].probe_loss(self.global_model) for c in active],
            dtype=float)
        self._last_losses = losses.tolist()
        rr = self.risk_scorer.score(
            active, losses, flat, self.client_sizes[active], hist)

        # --- safety gate ---
        # `gate_mode` decides where the reliability signal is allowed to act.
        # In "reward_only" the training channel is untouched plain FedDyn over
        # every active client, and risk governs payment eligibility alone. That
        # is the configuration the measurements support: the aggregation gate
        # costs accuracy on 8 of 9 conditions, while the same scores are
        # genuinely useful for deciding who gets paid.
        mode = str(getattr(cfg, "dcfi_gate_mode", "full"))
        if mode == "full":
            kept, kept_ids, actions = self.safety_gate.apply(
                global_state, client_states, rr, hist, round_idx)
        else:
            kept, kept_ids = list(client_states), list(active)
            actions = ["full"] * len(active)
            if mode == "reward_only":
                # Quarantine still applies to SELECTION (a repeatedly bad client
                # stops being invited), but never removes an update mid-round.
                for cid, tier in zip(rr.ids, rr.tier):
                    h = hist[cid]
                    if int(tier) == HIGH:
                        h.high_risk_streak += 1
                        if h.high_risk_streak >= self.safety_gate.quarantine_after:
                            h.quarantined_until = (
                                round_idx + self.safety_gate.quarantine_rounds)
                    else:
                        h.high_risk_streak = 0
        for cid, tier in zip(rr.ids, rr.tier):
            update_reliability(hist[cid], int(tier))

        # Payment eligibility: in reward_only mode the risk tier decides who is
        # PAID, decoupled from who is aggregated. HIGH earns nothing; MEDIUM is
        # paid at a reduced rate rather than zero, since a soft flag on five
        # clients is weak evidence.
        pay_scale = np.ones(len(active))
        if mode == "reward_only":
            pay_scale = np.array([{0: 1.0, 1: 0.5, 2: 0.0}[int(t)] for t in rr.tier])

        if not kept:                       # everyone excluded: hold the model
            new_state = {k: v.clone() for k, v in global_state.items()}
            weights = np.zeros(0)
        else:
            for cid in kept_ids:
                hist[cid].accepted += 1
            # UNIFORM weights: this is the whole point. Reward never becomes an
            # aggregation weight, so paying a client less cannot hurt the model.
            weights = np.ones(len(kept)) / len(kept)
            new_state = self._agg_feddyn(global_state, kept, weights=weights)

        # --- reward channel (off the training path) ---
        utility, fairness = self._dcfi_utility(global_state, kept, kept_ids, new_state)
        for cid, u in zip(kept_ids, utility):
            h = hist[cid]
            h.utility_ema = 0.7 * h.utility_ema + 0.3 * float(u)
        # Apply payment eligibility (reward_only mode); in "full" mode this is a
        # no-op because the gate has already removed high-risk clients from
        # `kept_ids` altogether.
        pay_u, pay_f = np.asarray(utility, float).copy(), np.asarray(fairness, float).copy()
        if mode == "reward_only" and len(pay_scale) == len(active):
            scale_by_id = dict(zip(active, pay_scale))
            s = np.array([scale_by_id.get(c, 1.0) for c in kept_ids])
            pay_u = np.where(s > 0, pay_u * s, -np.inf)
            pay_f = np.where(s > 0, pay_f * s, -np.inf)
        eligible = [i for i in range(len(kept_ids)) if np.isfinite(pay_u[i])]
        paid = self.reward_channel.pay(
            [kept_ids[i] for i in eligible], pay_u[eligible], pay_f[eligible], hist)

        return dict(
            new_state=new_state, weights=weights, kept_ids=kept_ids,
            risk=rr.risk.tolist(), tier=[TIER_NAMES[int(t)] for t in rr.tier],
            risk_components={k: v.tolist() for k, v in rr.components.items()},
            actions=actions, utility=np.asarray(utility).tolist(),
            fairness=np.asarray(fairness).tolist(),
            reward={int(k): float(v) for k, v in paid.items()},
            reliability={int(c): float(hist[c].reliability) for c in active},
            quarantined=[int(c) for c in hist if hist[c].quarantined_until >= round_idx],
        )

    @torch.no_grad()
    def _dcfi_utility(self, global_state, kept, kept_ids, full_state):
        """Approximate leave-one-out utility and fairness contribution.

        For each accepted client we re-aggregate WITHOUT it and measure how much
        validation balanced accuracy (utility) and minority-class recall
        (fairness) fall. This is the within-round LOO the framework calls for,
        and it costs |kept| cheap evaluations on the validation split.
        """
        n = len(kept)
        if n == 0:
            return np.zeros(0), np.zeros(0)
        if n == 1:
            base = self._val_metrics(full_state)
            return np.array([base["bacc"]]), np.array([base["minority_recall"]])

        full = self._val_metrics(full_state)
        util, fair = [], []
        for i in range(n):
            others = [cs for j, cs in enumerate(kept) if j != i]
            w = np.ones(len(others)) / len(others)
            # Plain weighted average, not the FedDyn update: we want this
            # client's marginal effect on the aggregate, not a counterfactual
            # server state (which would perturb the shared h term).
            s = aggregate(others, w)
            m = self._val_metrics(s)
            util.append(full["bacc"] - m["bacc"])
            fair.append(full["minority_recall"] - m["minority_recall"])
        return np.array(util), np.array(fair)

    @torch.no_grad()
    def _val_metrics(self, state) -> Dict[str, float]:
        """Balanced accuracy + minority-class recall on the validation split."""
        cfg = self.cfg
        model = self.model_builder().to(cfg.device)
        model.load_state_dict(state)
        model.eval()
        C = cfg.num_classes
        correct = np.zeros(C)
        total = np.zeros(C)
        for x, y in self.val_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            pred = model(x).argmax(dim=1)
            for c in range(C):
                m = (y == c)
                total[c] += m.sum().item()
                correct[c] += (pred[m] == c).sum().item()
        seen = total > 0
        recall = np.where(seen, correct / np.maximum(total, 1), 0.0)
        bacc = float(recall[seen].mean()) if seen.any() else 0.0
        # Minority class = rarest present class in the validation split.
        minority = int(np.argmin(np.where(seen, total, np.inf)))
        return {"bacc": bacc, "minority_recall": float(recall[minority])}

    def _diagnostics(self, global_state, client_states, cssv) -> Dict:
        """Per-client update norm + CSSV score, logged every round for every
        method. The norm is what exposes an inflated (f4) update, which cosine
        similarity is scale-invariant to and therefore cannot see."""
        out: Dict = {}
        try:
            norms = []
            for cs in client_states:
                tot = 0.0
                for k, g in global_state.items():
                    if torch.is_floating_point(g):
                        tot += float(((cs[k].float() - g.float()) ** 2).sum())
                norms.append(tot ** 0.5)
            out["update_norms"] = norms
        except Exception:                     # diagnostics must never break a run
            out["update_norms"] = None
        try:
            c = cssv if cssv is not None else compute_cssv(
                global_state=global_state, client_states=client_states,
                classifier_weight_key=CLASSIFIER_WEIGHT_KEY,
                num_classes=self.cfg.num_classes,
                reference=getattr(self.cfg, "cssv_reference", "mean"),
                trim_frac=getattr(self.cfg, "cssv_trim_frac", 0.2),
            )
            out["diag_cssv_score"] = ((np.asarray(c) + 1.0) / 2.0).sum(axis=1).tolist()
        except Exception:
            out["diag_cssv_score"] = None
        return out

    # ---------- aggregation helpers ----------

    def _shapfed_weights(self, global_state, client_states, active_sizes, active, round_idx):
        """Compute Shapley (CSSV) aggregation weights + update reputation EMA.
        Shared by `shapfed_wa` (plain weighted avg) and `shapfed_dyn` (weights
        feed FedDyn's server update)."""
        cfg = self.cfg
        cssv = compute_cssv(
            global_state=global_state,
            client_states=client_states,
            classifier_weight_key=CLASSIFIER_WEIGHT_KEY,
            num_classes=cfg.num_classes,
            reference=getattr(cfg, "cssv_reference", "mean"),
            trim_frac=getattr(cfg, "cssv_trim_frac", 0.2),
        )
        if cfg.cssv_unit_interval:
            per_client_score = ((cssv + 1.0) / 2.0).sum(axis=1)
        else:
            per_client_score = cssv.sum(axis=1)

        prev = None
        if cfg.cssv_ema > 0.0 and self.cssv_ema is not None:
            prev = self.cssv_ema[active]

        # Anneal blend lambda over rounds.
        t = round_idx / max(cfg.num_rounds - 1, 1)
        lam = (1 - t) * cfg.agg_blend_lambda + t * cfg.agg_blend_lambda_final

        weights = cssv_to_weights(
            cssv=cssv,
            client_sizes=active_sizes,
            temperature=cfg.cssv_temperature,
            clamp_negative=cfg.cssv_clamp_negative,
            ema_prev=prev,
            ema_momentum=cfg.cssv_ema,
            size_prior=True,
            blend_lambda=lam,
            max_weight=cfg.cssv_max_weight,
            unit_interval=cfg.cssv_unit_interval,
        )

        if self.cssv_ema is None:
            self.cssv_ema = np.zeros(len(self.clients))
        self.cssv_ema[active] = (
            cfg.cssv_ema * self.cssv_ema[active]
            + (1 - cfg.cssv_ema) * per_client_score
        )
        return weights, cssv

    def _loss_weights(self, active_sizes, active, round_idx):
        """Reliable label-noise detector: weight each active client by the global
        model's loss on its OWN data. Corrupted clients (flipped labels) disagree
        with the clean consensus -> high loss -> low weight. Blended with the size
        prior and capped, exactly like the CSSV path, and the per-client loss
        feeds the same reputation EMA (as a negative score)."""
        cfg = self.cfg
        losses = np.array([
            self.clients[cid].probe_loss(self.global_model) for cid in active
        ], dtype=np.float64)
        self._last_losses = losses.tolist()

        # exp(-beta * (loss - min_loss)) -> high-loss (noisy) clients get ~0.
        z = -float(cfg.loss_weight_beta) * (losses - losses.min())
        shap_w = np.exp(z)
        shap_w = shap_w / shap_w.sum() if shap_w.sum() > 0 else np.ones_like(shap_w) / len(shap_w)

        size_w = active_sizes / active_sizes.sum()
        t = round_idx / max(cfg.num_rounds - 1, 1)
        lam = (1 - t) * cfg.agg_blend_lambda + t * cfg.agg_blend_lambda_final
        w = (1.0 - lam) * size_w + lam * shap_w
        w = w / w.sum()
        w = _cap_weights(w, cfg.cssv_max_weight)

        # reputation EMA on the (negated, normalized) loss, for reputation-aware selection
        rep = 1.0 - _minmax(losses)
        if self.cssv_ema is None:
            self.cssv_ema = np.zeros(len(self.clients))
        self.cssv_ema[active] = cfg.cssv_ema * self.cssv_ema[active] + (1 - cfg.cssv_ema) * rep
        return w

    def _agg_shapfed(self, global_state, client_states, active_sizes, active, round_idx):
        weights, cssv = self._shapfed_weights(
            global_state, client_states, active_sizes, active, round_idx
        )
        new_state = aggregate(client_states, weights)
        return new_state, weights, cssv

    def _update_scaffold_control(self, auxes):
        N = len(self.clients)
        S = len(auxes)
        if S == 0 or "delta_c" not in auxes[0]:
            return
        keys = list(auxes[0]["delta_c"].keys())
        if self.scaffold_c is None:
            self.scaffold_c = {k: torch.zeros_like(auxes[0]["delta_c"][k]) for k in keys}
        for k in keys:
            mean_delta = torch.stack([a["delta_c"][k] for a in auxes], 0).mean(0)
            self.scaffold_c[k] = self.scaffold_c[k] + (S / N) * mean_delta

    def _agg_feddyn(self, global_state, client_states, weights=None) -> Dict[str, torch.Tensor]:
        """FedDyn server update. ``weights`` (default uniform) sets the client
        average — the proposed method passes Shapley weights here so the dynamic
        regularization is combined with contribution-aware aggregation."""
        cfg = self.cfg
        alpha = float(cfg.feddyn_alpha)
        N = len(self.clients)
        if weights is None:
            weights = np.ones(len(client_states)) / len(client_states)
        avg = aggregate(client_states, weights)  # (weighted) mean of client models
        if self.feddyn_h is None:
            self.feddyn_h = {
                k: torch.zeros_like(v.float())
                for k, v in global_state.items() if torch.is_floating_point(v)
            }
        S = len(client_states)
        weight_consistent = bool(getattr(cfg, "feddyn_weight_consistent", True))
        out: Dict[str, torch.Tensor] = {}
        for k, ref in avg.items():
            if not torch.is_floating_point(ref) or _is_running_stat(k):
                out[k] = ref.clone()  # BN running stats: plain average, no h term
                continue
            g = global_state[k].float()
            if weight_consistent:
                # Track the drift of exactly the weighted average that forms the
                # model: Σ_k w_k (θ_k − g) = avg − g, scaled by participation S/N.
                # With uniform w this reduces to (1/N)Σ(θ_k − g), i.e. the legacy
                # update — so plain feddyn is unchanged; only shapfed_dyn differs.
                weighted_delta = avg[k].float() - g
                self.feddyn_h[k] = self.feddyn_h[k] - alpha * (S / N) * weighted_delta
            else:
                sum_delta = torch.zeros_like(ref, dtype=torch.float32)
                for st in client_states:
                    sum_delta += st[k].float() - g
                self.feddyn_h[k] = self.feddyn_h[k] - (alpha / N) * sum_delta
            out[k] = (avg[k].float() - (1.0 / alpha) * self.feddyn_h[k]).to(ref.dtype)
        return out

    def _store_client_bn(self, active, client_states):
        keys = bn_keys(client_states[0])
        for cid, st in zip(active, client_states):
            self.client_bn[cid] = {k: st[k].clone() for k in keys}

    def _apply_server_momentum(self, global_state, new_state):
        """Server-side momentum as an EMA of the round update (unit steady-state
        gain): v <- beta*v + (1-beta)*delta; w <- w_prev + v. This smooths the
        global trajectory across non-IID rounds WITHOUT the 1/(1-beta) step
        amplification of the accumulation form (which diverges at beta=0.9)."""
        beta = float(getattr(self.cfg, "server_momentum", 0.0))
        if beta <= 0.0:
            return new_state
        if self.momentum_buf is None:
            self.momentum_buf = {
                k: torch.zeros_like(v.float())
                for k, v in new_state.items() if torch.is_floating_point(v)
            }
        out: Dict[str, torch.Tensor] = {}
        for k, ref in new_state.items():
            if not torch.is_floating_point(ref) or _is_running_stat(k):
                out[k] = ref.clone()  # BN running stats: keep the averaged value
                continue
            delta = ref.float() - global_state[k].float()
            self.momentum_buf[k] = beta * self.momentum_buf[k] + (1.0 - beta) * delta
            out[k] = (global_state[k].float() + self.momentum_buf[k]).to(ref.dtype)
        return out

    # ---------- global-test evaluation ----------

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if getattr(self.cfg, "task", "single_label") == "multi_label":
            return self._evaluate_multilabel()
        return self._evaluate_single_label()

    @torch.no_grad()
    def _evaluate_single_label(self) -> Dict[str, float]:
        cfg = self.cfg
        self.global_model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        per_class_correct = np.zeros(cfg.num_classes)
        per_class_total = np.zeros(cfg.num_classes)
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_pos_prob: List[float] = []
        ce = nn.CrossEntropyLoss(reduction="sum")
        for x, y in self.test_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = self.global_model(x)
            loss_sum += ce(logits, y).item()
            if cfg.num_classes == 2:
                all_pos_prob.extend(
                    torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                )
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            for c in range(cfg.num_classes):
                m = (y == c)
                per_class_total[c] += m.sum().item()
                per_class_correct[c] += (pred[m] == c).sum().item()
        per_class_acc = np.where(
            per_class_total > 0, per_class_correct / np.maximum(per_class_total, 1), 0.0
        )
        out = {
            "loss": loss_sum / max(total, 1),
            "acc": correct / max(total, 1),
            "per_class_acc": per_class_acc.tolist(),
            # Mean per-class recall. The honest headline whenever the test set is
            # not balanced (e.g. the depression target is 58.6% positive, so plain
            # accuracy rewards predicting the majority class).
            "balanced_acc": float(per_class_acc[per_class_total > 0].mean())
            if (per_class_total > 0).any() else 0.0,
        }
        if cfg.num_classes == 2 and all_pos_prob:
            y = np.array(all_labels)
            p = np.array(all_pos_prob)
            out["auc"] = _binary_auc(y, p)
            out.update(_binary_quality_metrics(y, p, np.array(all_preds)))
        if cfg.dataset == "aptos":
            try:
                from sklearn.metrics import cohen_kappa_score
                out["qwk"] = float(
                    cohen_kappa_score(all_labels, all_preds, weights="quadratic")
                )
            except ImportError:
                pass
        return out

    @torch.no_grad()
    def _evaluate_multilabel(self) -> Dict[str, float]:
        """Multi-label eval (ChestX-ray14): BCE loss + macro AUC / mAP."""
        cfg = self.cfg
        self.global_model.eval()
        bce = nn.BCEWithLogitsLoss(reduction="sum")
        loss_sum, n_elem = 0.0, 0
        all_probs: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        for x, y in self.test_loader:
            x, y = x.to(cfg.device), y.to(cfg.device).float()
            logits = self.global_model(x)
            loss_sum += bce(logits, y).item()
            n_elem += y.numel()
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(y.cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        preds = (probs >= 0.5).astype(np.float32)

        per_class_acc = (preds == targets).mean(axis=0)
        out: Dict[str, float] = {
            "loss": loss_sum / max(n_elem, 1),
            "acc": float(per_class_acc.mean()),
            "per_class_acc": per_class_acc.tolist(),
        }
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            aucs, aps = [], []
            for c in range(targets.shape[1]):
                yc = targets[:, c]
                if yc.min() == yc.max():
                    continue
                aucs.append(roc_auc_score(yc, probs[:, c]))
                aps.append(average_precision_score(yc, probs[:, c]))
            if aucs:
                out["auc"] = float(np.mean(aucs))
                out["map"] = float(np.mean(aps))
                out["per_class_auc"] = aucs
        except ImportError:
            pass
        return out

    # ---------- per-client (fairness) evaluation ----------

    @torch.no_grad()
    def evaluate_per_client(self) -> Dict[str, float]:
        """Evaluate the global model on each client's local test set and reduce to
        Jain's fairness index. For FedBN the client's own BatchNorm is injected.
        Returns {} when no client has a local test set."""
        cfg = self.cfg
        multi_label = getattr(cfg, "task", "single_label") == "multi_label"
        global_state = None
        if cfg.aggregation == "fedbn":
            global_state = self.global_model.state_dict()

        accs: List[float] = []
        for c in self.clients:
            if c.test_loader is None:
                continue
            model = self.global_model
            if cfg.aggregation == "fedbn" and c.id in self.client_bn:
                model = self.model_builder().to(cfg.device)
                st = {k: v.clone() for k, v in global_state.items()}
                st.update(self.client_bn[c.id])
                model.load_state_dict(st)
            model.eval()
            accs.append(self._client_acc(model, c.test_loader, multi_label))

        if not accs:
            return {}
        a = np.array(accs, dtype=np.float64)
        jain = float((a.sum() ** 2) / (len(a) * (np.square(a).sum()) + 1e-12))
        return {
            "jain": jain,
            "client_acc_mean": float(a.mean()),
            "client_acc_min": float(a.min()),
            "client_acc_std": float(a.std()),
            "per_client_acc": a.tolist(),
        }

    @torch.no_grad()
    def _client_acc(self, model, loader, multi_label: bool) -> float:
        cfg = self.cfg
        if multi_label:
            correct, total = 0.0, 0
            for x, y in loader:
                x, y = x.to(cfg.device), y.to(cfg.device).float()
                pred = (torch.sigmoid(model(x)) >= 0.5).float()
                correct += (pred == y).float().sum().item()
                total += y.numel()
            return correct / max(total, 1)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)

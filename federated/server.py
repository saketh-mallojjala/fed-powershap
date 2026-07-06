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
from federated.selection import SelectionOutcome, select_clients
from federated.shapley import compute_cssv, cssv_to_weights
from models import CLASSIFIER_LAYER_NAME


CLASSIFIER_WEIGHT_KEY = f"{CLASSIFIER_LAYER_NAME}.weight"


def _is_running_stat(key: str) -> bool:
    """BN running buffers (mean/var/count) must be averaged, never momentum- or
    correction-updated: pushing running_var negative blows up BatchNorm."""
    return key.endswith(_BN_MARKERS)


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

    # ---------- selection ----------

    def _select(self, round_idx: int) -> SelectionOutcome:
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

        # ----- compute weights + new global state, per method -----
        if cfg.aggregation == "shapfed_wa":
            new_state, weights, cssv = self._agg_shapfed(
                global_state, client_states, active_sizes, outcome.active, round_idx
            )
        elif cfg.aggregation == "shapfed_dyn":
            # Proposed: contribution-aware (Shapley-weighted) FedDyn aggregation.
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

        return {
            "round": round_idx,
            "candidates": outcome.candidates,
            "candidate_losses": outcome.candidate_losses,
            "active": outcome.active,
            "weights": np.asarray(weights).tolist(),
            "cssv": cssv.tolist() if cssv is not None else None,
        }

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
        ce = nn.CrossEntropyLoss(reduction="sum")
        for x, y in self.test_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = self.global_model(x)
            loss_sum += ce(logits, y).item()
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
        }
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

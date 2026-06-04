"""Server: drives the round, wires Power-of-Choice selection into ShapFed."""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import get_test_loader
from federated.aggregation import aggregate
from federated.client import Client
from federated.selection import select_clients
from federated.shapley import compute_cssv, cssv_to_weights
from models import CLASSIFIER_LAYER_NAME


CLASSIFIER_WEIGHT_KEY = f"{CLASSIFIER_LAYER_NAME}.weight"


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
        self.cssv_ema: Optional[np.ndarray] = None  # smoothed per-client score
        self.history: List[Dict] = []

    # ---------- core round ----------

    def run_round(self, round_idx: int) -> Dict:
        cfg = self.cfg
        global_state = {
            k: v.detach().cpu().clone()
            for k, v in self.global_model.state_dict().items()
        }

        # 1. Power-of-Choice selection.
        def probe(cid: int) -> float:
            # Temporarily load global state into a fresh model on device
            # (the server's model already has it; reuse).
            return self.clients[cid].probe_loss(self.global_model)

        outcome = select_clients(
            strategy=cfg.selection_strategy,
            num_clients=len(self.clients),
            client_sizes=self.client_sizes,
            d=cfg.candidate_size_d,
            m=cfg.active_size_m,
            rng=self.rng,
            loss_fn=probe if cfg.selection_strategy == "pow_d" else None,
            size_weighted_candidates=cfg.size_weighted_candidates,
        )

        # 2. Local training on active clients.
        client_states: List[Dict[str, torch.Tensor]] = []
        for cid in outcome.active:
            client_states.append(
                self.clients[cid].local_train(global_state, self.model_builder)
            )

        # 3. Compute weights: ShapFed CSSV or FedAvg.
        active_sizes = self.client_sizes[outcome.active]

        if cfg.aggregation == "shapfed_wa":
            cssv = compute_cssv(
                global_state=global_state,
                client_states=client_states,
                classifier_weight_key=CLASSIFIER_WEIGHT_KEY,
                num_classes=cfg.num_classes,
            )
            # EMA is kept per *client id* so it carries across rounds even when
            # the active set changes. Build/align by client id.
            per_client_score = cssv.sum(axis=1)
            prev = None
            if cfg.cssv_ema > 0.0 and self.cssv_ema is not None:
                prev = self.cssv_ema[outcome.active]

            weights = cssv_to_weights(
                cssv=cssv,
                client_sizes=active_sizes,
                temperature=cfg.cssv_temperature,
                clamp_negative=cfg.cssv_clamp_negative,
                ema_prev=prev,
                ema_momentum=cfg.cssv_ema,
                size_prior=True,
            )
            # Update EMA store.
            if self.cssv_ema is None:
                self.cssv_ema = np.zeros(len(self.clients))
            self.cssv_ema[outcome.active] = (
                cfg.cssv_ema * self.cssv_ema[outcome.active]
                + (1 - cfg.cssv_ema) * per_client_score
            )
        elif cfg.aggregation == "fedavg":
            weights = active_sizes / active_sizes.sum()
            cssv = None
        else:
            raise ValueError(f"Unknown aggregation: {cfg.aggregation}")

        # 4. Aggregate and load into the global model.
        new_state = aggregate(client_states, weights)
        self.global_model.load_state_dict(new_state)

        return {
            "round": round_idx,
            "candidates": outcome.candidates,
            "candidate_losses": outcome.candidate_losses,
            "active": outcome.active,
            "weights": weights.tolist(),
            "cssv": cssv.tolist() if cssv is not None else None,
        }

    # ---------- evaluation ----------

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
        # Quadratic Weighted Kappa — APTOS' official metric. Cheap to compute,
        # sklearn import is local so non-aptos runs don't pay the import cost.
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
        """Multi-label eval (ChestX-ray14): BCE loss + macro AUC / mAP.

        ``acc`` is reported as the mean per-class accuracy at a 0.5 threshold so
        the metric keys stay compatible with the single-label plots; ``auc`` is
        the headline metric. ``per_class_acc`` (thresholded) and ``per_class_auc``
        are both logged.
        """
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

        probs = np.concatenate(all_probs, axis=0)        # (N, C)
        targets = np.concatenate(all_targets, axis=0)    # (N, C) in {0,1}
        preds = (probs >= 0.5).astype(np.float32)

        per_class_acc = (preds == targets).mean(axis=0)
        out: Dict[str, float] = {
            "loss": loss_sum / max(n_elem, 1),
            "acc": float(per_class_acc.mean()),
            "per_class_acc": per_class_acc.tolist(),
        }

        # Macro AUC / mAP — skip classes with a single label value in the test set.
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

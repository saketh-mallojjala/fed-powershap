"""Per-client state: holds data loaders, runs a local solver, reports loss.

The local solver is selected by ``cfg.local_solver``:
    sgd      - plain local SGD (FedAvg, FedBN, PoC-FedAvg, FedCE, Proposed)
    fedprox  - + proximal term  (mu/2)||w - w_global||^2          (Li et al. 2020)
    feddyn   - dynamic regularization with per-client gradient state (Acar 2021)
    moon     - model-contrastive loss vs. global (pos) / prev-local (neg) (Li 2021)
    scaffold - control-variate corrected local steps                (Karimireddy 2020)

``local_train`` returns ``(state_dict, aux)``; ``aux`` carries solver-specific
extras (e.g. SCAFFOLD's control delta) for the server to consume.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from data.dataset import make_client_loader
from federated.selection import evaluate_local_loss


class Client:
    def __init__(
        self,
        client_id: int,
        subset: Subset,
        labels: np.ndarray,
        cfg,
        test_subset: Optional[Subset] = None,
    ):
        self.id = client_id
        self.subset = subset
        self.labels = labels
        self.cfg = cfg
        nw = getattr(cfg, "num_workers", 0)
        self.train_loader = make_client_loader(
            subset, cfg.batch_size, shuffle=True, num_workers=nw
        )
        # A separate, non-shuffled loader used only for the pow-d loss probe.
        self.probe_loader = make_client_loader(
            subset, cfg.batch_size, shuffle=False, num_workers=nw
        )
        # Per-client local test set (for fairness / Jain index), or None.
        self.test_subset = test_subset
        self.test_loader = (
            make_client_loader(test_subset, cfg.batch_size, shuffle=False, num_workers=nw)
            if test_subset is not None and len(test_subset) > 0 else None
        )

        # Lazy per-client solver state.
        self._scaffold_c: Optional[Dict[str, torch.Tensor]] = None   # control variate
        self._feddyn_grad: Optional[Dict[str, torch.Tensor]] = None  # gradient memory
        self._moon_prev: Optional[Dict[str, torch.Tensor]] = None    # prev local model

    @property
    def size(self) -> int:
        return len(self.subset)

    @property
    def class_histogram(self) -> np.ndarray:
        hist = np.zeros(self.cfg.num_classes, dtype=np.int64)
        for c in self.labels:
            hist[int(c)] += 1
        return hist

    def probe_loss(self, global_model: nn.Module) -> float:
        return evaluate_local_loss(
            global_model, self.probe_loader, self.cfg.device, max_batches=4,
            task=getattr(self.cfg, "task", "single_label"),
        )

    # ---------- local training ----------

    def local_train(
        self,
        global_state: Dict[str, torch.Tensor],
        model_builder,
        server_control: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Run the configured local solver. Returns (new_state_dict, aux)."""
        cfg = self.cfg
        solver = getattr(cfg, "local_solver", "sgd")
        model = model_builder().to(cfg.device)
        model.load_state_dict(global_state)
        model.train()

        if solver == "scaffold":
            return self._train_scaffold(model, global_state, server_control)
        if solver == "moon":
            return self._train_moon(model, global_state, model_builder)
        if solver == "feddyn":
            return self._train_feddyn(model, global_state)
        # sgd or fedprox
        return self._train_sgd(model, global_state, solver)

    def _criterion(self):
        multi_label = getattr(self.cfg, "task", "single_label") == "multi_label"
        return (nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()), multi_label

    def _optimizer(self, model):
        cfg = self.cfg
        return torch.optim.SGD(
            model.parameters(), lr=cfg.local_lr,
            momentum=cfg.momentum, weight_decay=cfg.weight_decay,
        )

    @staticmethod
    def _state(model) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _train_sgd(self, model, global_state, solver) -> Tuple[Dict, Dict]:
        """Plain SGD (solver='sgd') or FedProx (solver='fedprox')."""
        cfg = self.cfg
        criterion, multi_label = self._criterion()
        optimizer = self._optimizer(model)
        mu = float(getattr(cfg, "fedprox_mu", 0.0)) if solver == "fedprox" else 0.0
        g_params = None
        if mu > 0.0:
            g_params = {n: global_state[n].to(cfg.device) for n, _ in model.named_parameters()}

        for _ in range(cfg.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                if multi_label:
                    y = y.float()
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                if mu > 0.0:
                    prox = sum(
                        ((p - g_params[n]) ** 2).sum()
                        for n, p in model.named_parameters()
                    )
                    loss = loss + 0.5 * mu * prox
                loss.backward()
                optimizer.step()

        return self._state(model), {}

    def _train_feddyn(self, model, global_state) -> Tuple[Dict, Dict]:
        """FedDyn (Acar et al. 2021): linear gradient memory + proximal term."""
        cfg = self.cfg
        alpha = float(getattr(cfg, "feddyn_alpha", 0.01))
        criterion, multi_label = self._criterion()
        optimizer = self._optimizer(model)
        named = dict(model.named_parameters())
        g_params = {n: global_state[n].to(cfg.device) for n in named}
        if self._feddyn_grad is None:
            self._feddyn_grad = {n: torch.zeros_like(g_params[n]) for n in named}
        h = {n: self._feddyn_grad[n].to(cfg.device) for n in named}

        for _ in range(cfg.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                if multi_label:
                    y = y.float()
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                # FedDyn regularizer: -<h, w> + (alpha/2)||w - w_global||^2
                lin = sum((-(h[n] * p).sum()) for n, p in model.named_parameters())
                quad = sum(((p - g_params[n]) ** 2).sum() for n, p in model.named_parameters())
                loss = loss + lin + 0.5 * alpha * quad
                loss.backward()
                optimizer.step()

        # Update gradient memory: h <- h - alpha (w_local - w_global)
        with torch.no_grad():
            for n, p in model.named_parameters():
                h[n] = h[n] - alpha * (p.detach() - g_params[n])
            self._feddyn_grad = {n: h[n].detach().cpu().clone() for n in named}

        return self._state(model), {}

    def _train_moon(self, model, global_state, model_builder) -> Tuple[Dict, Dict]:
        """MOON (Li et al. 2021): model-contrastive regularization."""
        cfg = self.cfg
        mu = float(getattr(cfg, "moon_mu", 1.0))
        tau = float(getattr(cfg, "moon_temperature", 0.5))
        criterion, multi_label = self._criterion()
        optimizer = self._optimizer(model)

        global_model = model_builder().to(cfg.device)
        global_model.load_state_dict(global_state)
        global_model.eval()
        prev_model = model_builder().to(cfg.device)
        prev_model.load_state_dict(self._moon_prev if self._moon_prev is not None else global_state)
        prev_model.eval()

        ce_contrast = nn.CrossEntropyLoss()
        for _ in range(cfg.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                if multi_label:
                    y = y.float()
                optimizer.zero_grad()
                feat = model.extract_features(x)
                logits = model.classifier(feat)
                loss = criterion(logits, y)
                with torch.no_grad():
                    g_feat = global_model.extract_features(x)
                    p_feat = prev_model.extract_features(x)
                pos = F.cosine_similarity(feat, g_feat, dim=1)   # to global (+)
                neg = F.cosine_similarity(feat, p_feat, dim=1)   # to prev local (-)
                con_logits = torch.stack([pos, neg], dim=1) / tau
                con_labels = torch.zeros(x.size(0), dtype=torch.long, device=cfg.device)
                loss = loss + mu * ce_contrast(con_logits, con_labels)
                loss.backward()
                optimizer.step()

        state = self._state(model)
        self._moon_prev = {k: v.clone() for k, v in state.items()}  # for next round
        return state, {}

    def _train_scaffold(self, model, global_state, server_control) -> Tuple[Dict, Dict]:
        """SCAFFOLD (Karimireddy et al. 2020): control-variate corrected SGD."""
        cfg = self.cfg
        criterion, multi_label = self._criterion()
        # SCAFFOLD corrects .grad directly, so it assumes plain SGD: momentum
        # would entangle the control correction with the momentum buffer and
        # break the variate accounting (causing divergence). Use momentum=0.
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.local_lr, momentum=0.0,
            weight_decay=cfg.weight_decay,
        )
        named = dict(model.named_parameters())
        if self._scaffold_c is None:
            self._scaffold_c = {n: torch.zeros_like(named[n], device="cpu") for n in named}
        c_i = {n: self._scaffold_c[n].to(cfg.device) for n in named}
        c = (
            {n: server_control[n].to(cfg.device) for n in named}
            if server_control is not None
            else {n: torch.zeros_like(named[n]) for n in named}
        )

        steps = 0
        for _ in range(cfg.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                if multi_label:
                    y = y.float()
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                # SCAFFOLD correction: g <- g + (c - c_i)
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            p.grad.add_(c[n] - c_i[n])
                optimizer.step()
                steps += 1

        # Update control variate (option II): c_i^+ = c_i - c + (w_g - w)/(K*lr)
        coef = 1.0 / max(steps * cfg.local_lr, 1e-12)
        delta_c: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                new_ci = c_i[n] - c[n] + coef * (global_state[n].to(cfg.device) - p.detach())
                delta_c[n] = (new_ci - c_i[n]).detach().cpu().clone()
                c_i[n] = new_ci
            self._scaffold_c = {n: c_i[n].detach().cpu().clone() for n in named}

        return self._state(model), {"delta_c": delta_c}

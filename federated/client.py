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
        lr: Optional[float] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Run the configured local solver. Returns (new_state_dict, aux).

        ``lr`` overrides ``cfg.local_lr`` for this round (server LR schedule)."""
        cfg = self.cfg
        self._cur_lr = float(lr) if lr is not None else float(cfg.local_lr)
        solver = getattr(cfg, "local_solver", "sgd")

        # Adversaries short-circuit or distort the honest solver. Those that
        # never train return immediately; the rest train first and corrupt the
        # result afterwards. See ATTACKS in federated/dcfi.py for the taxonomy.
        fr = getattr(self, "freerider_mode", "none")
        if fr in ("f2", "zero_update"):                   # zero effort
            return {k: v.clone() for k, v in global_state.items()}, {}
        if fr in ("f1", "random_update"):
            return self._attack_random(global_state), {}
        if fr == "scaled_global":
            return self._attack_scaled_global(global_state), {}
        # Replay attacks need something to replay. They train honestly ONCE to
        # seed the cache, then resend it for free forever after -- which is the
        # realistic threat model (pay the cost once, harvest indefinitely).
        # Returning early before ever training would make them degenerate into
        # zero_update, which is a different attack entirely.
        if fr in ("stale_model", "copied_update"):
            if getattr(self, "_last_state", None) is None:
                model = model_builder().to(cfg.device)
                model.load_state_dict(global_state)
                model.train()
                state, aux = self._dispatch_solver(
                    solver, model, global_state, model_builder, server_control)
                self._remember(global_state, state)
                return state, aux
            return ((self._attack_stale(global_state) if fr == "stale_model"
                     else self._attack_copied(global_state)), {})

        model = model_builder().to(cfg.device)
        model.load_state_dict(global_state)
        model.train()

        if fr in ("f3", "low_effort"):
            state, aux = self._train_sgd(model, global_state, solver,
                                         loader=self._subset_loader())
        else:
            state, aux = self._dispatch_solver(
                solver, model, global_state, model_builder, server_control)

        if fr in ("f4", "norm_inflation"):
            state = self._inflate(global_state, state)

        self._remember(global_state, state)
        return state, aux

    def _dispatch_solver(self, solver, model, global_state, model_builder,
                         server_control):
        if solver == "scaffold":
            return self._train_scaffold(model, global_state, server_control)
        if solver == "moon":
            return self._train_moon(model, global_state, model_builder)
        if solver == "feddyn":
            return self._train_feddyn(model, global_state)
        # sgd or fedprox
        return self._train_sgd(model, global_state, solver)

    # ---------- free-rider behaviours (proposal §9.3) ----------

    def _freerider_random(self, global_state: Dict[str, torch.Tensor]) -> Dict:
        """f1: a random update, scaled to be a *plausible* size.

        Sigma is tied to each tensor's own std rather than a fixed constant, so
        the fake update is not trivially detectable by raw magnitude alone --
        the detector has to notice it points nowhere useful, not that it is big.
        """
        scale = float(getattr(self.cfg, "freerider_noise_scale", 1.0))
        g = torch.Generator().manual_seed(hash((self.id, id(self))) % (2 ** 31))
        out = {}
        for k, v in global_state.items():
            if torch.is_floating_point(v) and v.numel() > 1:
                sigma = 0.01 * scale * float(v.detach().float().std().clamp(min=1e-8))
                noise = torch.randn(v.shape, generator=g, dtype=torch.float32)
                out[k] = (v.detach().float() + sigma * noise).to(v.dtype)
            else:
                out[k] = v.clone()
        return out

    def _remember(self, global_state, state):
        """Cache this round's honest update, so replay-style attacks
        (stale_model, copied_update) have something to resend."""
        self._last_delta = {
            k: (state[k].detach().float() - v.detach().float())
            for k, v in global_state.items()
            if torch.is_floating_point(v) and k in state
        }
        self._last_state = {k: v.detach().clone() for k, v in state.items()}

    def _attack_stale(self, global_state: Dict[str, torch.Tensor]) -> Dict:
        """Stale-model return: resend a model trained in an EARLIER round.

        Costs nothing after the first round and looks like a real trained model
        -- its probe loss is genuine, its direction is plausible. What gives it
        away is that it does not move with the global model, so its apparent
        update grows as the global model advances past it."""
        if getattr(self, "_last_state", None) is None:
            return {k: v.clone() for k, v in global_state.items()}
        return {k: (self._last_state[k].clone() if k in self._last_state else v.clone())
                for k, v in global_state.items()}

    def _attack_scaled_global(self, global_state: Dict[str, torch.Tensor]) -> Dict:
        """Scaled global update: return c * w_global without training at all.

        Produces a large-norm update aligned with nothing in particular; the
        naive 'bigger update = more work' heuristic rewards it."""
        c = float(getattr(self.cfg, "freerider_scale", 3.0))
        return {k: ((c * v.detach().float()).to(v.dtype)
                    if torch.is_floating_point(v) else v.clone())
                for k, v in global_state.items()}

    def _attack_copied(self, global_state: Dict[str, torch.Tensor]) -> Dict:
        """Copied update: replay this client's own previous delta on top of the
        current global model. Stands in for copying a peer's update -- the
        server sees a well-formed, correctly-directed update that cost nothing
        to produce this round."""
        if getattr(self, "_last_delta", None) is None:
            return {k: v.clone() for k, v in global_state.items()}
        return {k: ((v.detach().float() + self._last_delta[k]).to(v.dtype)
                    if k in self._last_delta else v.clone())
                for k, v in global_state.items()}

    def _attack_random(self, global_state: Dict[str, torch.Tensor]) -> Dict:
        return self._freerider_random(global_state)

    def _subset_loader(self):
        """f3: a loader over only `freerider_subset_frac` of the local data."""
        if getattr(self, "_fr_loader", None) is None:
            frac = float(getattr(self.cfg, "freerider_subset_frac", 0.1))
            n = max(1, int(round(len(self.subset) * frac)))
            rng = np.random.default_rng(1000 + self.id)
            idx = rng.choice(len(self.subset), size=n, replace=False).tolist()
            self._fr_loader = make_client_loader(
                Subset(self.subset, idx), self.cfg.batch_size, shuffle=True,
                num_workers=getattr(self.cfg, "num_workers", 0),
            )
        return self._fr_loader

    def _inflate(self, global_state: Dict[str, torch.Tensor],
                 state: Dict[str, torch.Tensor]) -> Dict:
        """f4: scale the honest update by c>1 to inflate apparent contribution.

        Cosine-based signals are scale-invariant, so this attack is *invisible*
        to CSSV by construction; it shows up only in the update norm. That
        contrast is the point of including it in the ladder.
        """
        c = float(getattr(self.cfg, "freerider_scale", 3.0))
        out = {}
        for k, v in global_state.items():
            if torch.is_floating_point(v) and k in state:
                out[k] = (v.detach().float()
                          + c * (state[k].detach().float() - v.detach().float())).to(v.dtype)
            else:
                out[k] = state.get(k, v).clone()
        return out

    def _criterion(self):
        multi_label = getattr(self.cfg, "task", "single_label") == "multi_label"
        return (nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()), multi_label

    def _optimizer(self, model):
        cfg = self.cfg
        return torch.optim.SGD(
            model.parameters(), lr=getattr(self, "_cur_lr", cfg.local_lr),
            momentum=cfg.momentum, weight_decay=cfg.weight_decay,
        )

    @staticmethod
    def _state(model) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _train_sgd(self, model, global_state, solver, loader=None) -> Tuple[Dict, Dict]:
        """Plain SGD (solver='sgd') or FedProx (solver='fedprox').

        ``loader`` overrides the client's own training loader (used by the f3
        free-rider, which trains on only a fraction of its data)."""
        cfg = self.cfg
        criterion, multi_label = self._criterion()
        optimizer = self._optimizer(model)
        mu = float(getattr(cfg, "fedprox_mu", 0.0)) if solver == "fedprox" else 0.0
        g_params = None
        if mu > 0.0:
            g_params = {n: global_state[n].to(cfg.device) for n, _ in model.named_parameters()}

        for _ in range(cfg.local_epochs):
            for x, y in (loader if loader is not None else self.train_loader):
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
        lr = getattr(self, "_cur_lr", cfg.local_lr)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.0,
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
        coef = 1.0 / max(steps * lr, 1e-12)
        delta_c: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                new_ci = c_i[n] - c[n] + coef * (global_state[n].to(cfg.device) - p.detach())
                delta_c[n] = (new_ci - c_i[n]).detach().cpu().clone()
                c_i[n] = new_ci
            self._scaffold_c = {n: c_i[n].detach().cpu().clone() for n in named}

        return self._state(model), {"delta_c": delta_c}

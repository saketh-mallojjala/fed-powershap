"""Per-client state: holds a data loader, runs local SGD, reports loss."""
from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
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
            global_model, self.probe_loader, self.cfg.device, max_batches=4
        )

    def local_train(
        self,
        global_state: Dict[str, torch.Tensor],
        model_builder,
    ) -> Dict[str, torch.Tensor]:
        """Load global weights, run local SGD, return the new state_dict."""
        cfg = self.cfg
        model = model_builder().to(cfg.device)
        model.load_state_dict(global_state)
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.local_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        for _ in range(cfg.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

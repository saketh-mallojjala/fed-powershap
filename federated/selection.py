"""Client selection strategies, including Power-of-Choice (pow-d).

Power-of-Choice (Cho et al., 2020):
    1. Sample a candidate pool A_t of size d, with probability proportional
       to the local dataset size |D_k| / |D|.
    2. Each candidate reports its local loss F_k(w_t) evaluated on the
       current global model.
    3. Select the m clients in A_t with the largest local loss.

This module deliberately returns both the candidate set and the active set so
callers (and logs) can inspect what Power-of-Choice considered vs. trained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch


@dataclass
class SelectionOutcome:
    candidates: List[int]
    active: List[int]
    candidate_losses: Optional[List[float]] = None  # aligned with `candidates`


def _sample_candidates(
    num_clients: int,
    d: int,
    client_sizes: np.ndarray,
    size_weighted: bool,
    rng: np.random.Generator,
) -> List[int]:
    d = min(d, num_clients)
    if size_weighted:
        p = client_sizes / client_sizes.sum()
    else:
        p = np.ones(num_clients) / num_clients
    idx = rng.choice(num_clients, size=d, replace=False, p=p)
    return idx.tolist()


def select_clients(
    strategy: str,
    num_clients: int,
    client_sizes: np.ndarray,
    d: int,
    m: int,
    rng: np.random.Generator,
    loss_fn: Optional[Callable[[int], float]] = None,
    size_weighted_candidates: bool = True,
) -> SelectionOutcome:
    """Dispatch to a selection strategy.

    For `pow_d`, `loss_fn(client_id) -> float` must be provided. Each candidate
    is queried once per round on the current global model.
    """
    m = min(m, num_clients)

    if strategy == "full":
        clients = list(range(num_clients))
        return SelectionOutcome(candidates=clients, active=clients)

    if strategy == "random":
        p = client_sizes / client_sizes.sum() if size_weighted_candidates else None
        active = rng.choice(num_clients, size=m, replace=False, p=p).tolist()
        return SelectionOutcome(candidates=active, active=active)

    if strategy == "pow_d":
        if loss_fn is None:
            raise ValueError("pow_d selection requires a loss_fn callback")
        candidates = _sample_candidates(
            num_clients, d, client_sizes, size_weighted_candidates, rng
        )
        losses = [float(loss_fn(c)) for c in candidates]
        # Top-m highest loss.
        order = np.argsort(losses)[::-1]
        active = [candidates[i] for i in order[:m]]
        return SelectionOutcome(
            candidates=candidates, active=active, candidate_losses=losses
        )

    raise ValueError(f"Unknown selection strategy: {strategy}")


@torch.no_grad()
def evaluate_local_loss(
    model: torch.nn.Module,
    loader,
    device: str,
    max_batches: int = 4,
    task: str = "single_label",
) -> float:
    """Proxy for F_k(w_t): mean loss over a few mini-batches.

    Cross-entropy for single-label, BCE for multi-label. Power-of-Choice
    queries this per candidate per round, so cap the batch count to keep
    overhead small.
    """
    model.eval()
    multi_label = task == "multi_label"
    criterion = (
        torch.nn.BCEWithLogitsLoss(reduction="sum") if multi_label
        else torch.nn.CrossEntropyLoss(reduction="sum")
    )
    total_loss, total_n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if multi_label:
            y = y.float()
        logits = model(x)
        total_loss += criterion(logits, y).item()
        total_n += y.numel()
    return total_loss / max(total_n, 1)

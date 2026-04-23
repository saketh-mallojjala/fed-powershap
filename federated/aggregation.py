"""Weighted aggregation of client state_dicts."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


def aggregate(
    client_states: Sequence[Dict[str, torch.Tensor]],
    weights: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Return Σ_k w_k · client_state_k, keyed by parameter name.

    `weights` is assumed normalized (sums to 1). Parameters with non-floating
    dtypes (e.g. BN `num_batches_tracked`) are copied from the first client.
    """
    assert len(client_states) == len(weights), "weights / clients length mismatch"
    assert len(client_states) > 0, "cannot aggregate empty client list"

    keys = list(client_states[0].keys())
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        ref = client_states[0][key]
        if not torch.is_floating_point(ref):
            out[key] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for w, state in zip(weights, client_states):
            acc += float(w) * state[key].to(torch.float32)
        out[key] = acc.to(ref.dtype)
    return out

"""Lightweight JSON-lines logger and seed helper."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JsonLogger:
    def __init__(self, log_dir: str, exp_name: str):
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(log_dir, f"{exp_name}-{stamp}.jsonl")
        self.f = open(self.path, "w", buffering=1)

    def log(self, record: Dict[str, Any]) -> None:
        self.f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

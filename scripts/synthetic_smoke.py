"""End-to-end pipeline check on a synthetic 5-class dataset (no APTOS download).

Mirrors the APTOS configuration (ResNet-18 pretrained, 224x224, 5 classes,
ImageNet normalization) but feeds random tensors so we can exercise the full
ShapFed-WA + Power-of-Choice loop without disk I/O.

Catches: model build, FL aggregation key alignment, BN behavior under
weighted state-dict averaging, CSSV row-shape mismatch, QWK on a tiny eval
set, MPS/CUDA fallback.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch
from torch.utils.data import Dataset

from config import get_config
from federated import Client, Server
from models import build_model, CLASSIFIER_LAYER_NAME
from utils import set_seed


class SyntheticImageDataset(Dataset):
    """Random tensors with deterministic per-sample labels."""
    def __init__(self, n: int, num_classes: int, image_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.images = torch.from_numpy(
            rng.standard_normal((n, 3, image_size, image_size)).astype(np.float32)
        )
        self.targets = rng.integers(0, num_classes, size=n).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.images[i], int(self.targets[i])


def run():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    cfg = get_config(
        dataset="aptos",          # only used for transform branch — we override the data
        model="resnet18",
        pretrained=False,         # don't download ImageNet weights for this check
        num_classes=5,
        image_size=64,            # smaller than 224 to keep CPU/MPS fast
        num_clients=4,
        alpha=0.5,
        num_rounds=2,
        local_epochs=1,
        candidate_size_d=4,
        active_size_m=2,
        batch_size=8,
        num_workers=0,
        device=device,
        exp_name="synthetic_smoke",
    )
    set_seed(cfg.seed)
    print(f"[synthetic] device={cfg.device}")

    # Build dataset + manual partition (skip the data/dataset.py path).
    train_set = SyntheticImageDataset(n=120, num_classes=cfg.num_classes, image_size=cfg.image_size, seed=1)
    test_set = SyntheticImageDataset(n=40, num_classes=cfg.num_classes, image_size=cfg.image_size, seed=2)

    labels = np.array(train_set.targets)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(train_set))
    splits = np.array_split(perm, cfg.num_clients)
    from torch.utils.data import Subset
    subsets = [Subset(train_set, s.tolist()) for s in splits]
    client_labels = {k: labels[s] for k, s in enumerate(splits)}

    clients = [Client(i, subsets[i], client_labels[i], cfg) for i in range(cfg.num_clients)]
    server = Server(cfg, clients, test_set, lambda: build_model(cfg))

    # Verify ShapFed will find the classifier weights.
    sd = server.global_model.state_dict()
    key = f"{CLASSIFIER_LAYER_NAME}.weight"
    assert key in sd, f"missing classifier weight key {key} in state_dict"
    assert sd[key].shape == (cfg.num_classes, sd[key].shape[1]), \
        f"classifier weight should be (C, F); got {sd[key].shape}"
    print(f"[synthetic] classifier key={key} shape={tuple(sd[key].shape)}")

    for r in range(cfg.num_rounds):
        info = server.run_round(r)
        ws = [round(w, 3) for w in info["weights"]]
        cssv_shape = None if info["cssv"] is None else (len(info["cssv"]), len(info["cssv"][0]))
        print(f"round {r}: active={info['active']} weights={ws} cssv_shape={cssv_shape}")

    m = server.evaluate()
    print(
        f"eval acc={m['acc']:.3f} loss={m['loss']:.3f} "
        f"qwk={m.get('qwk', float('nan')):.3f}"
    )
    assert 0.0 <= m["acc"] <= 1.0
    assert "qwk" in m, "QWK not reported (sklearn missing?)"
    print("synthetic smoke OK")


if __name__ == "__main__":
    run()

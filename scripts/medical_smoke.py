"""Tiny end-to-end sanity check for the medical datasets: 5 clients, 2 rounds.

MedMNIST datasets (octmnist / isic / chestxray14) download in seconds at 28px,
so this is a fast wiring check for the loader, transforms, Dirichlet partition,
and (for chestxray14) the multi-label BCE / AUC path.

    python scripts/medical_smoke.py octmnist
    python scripts/medical_smoke.py chestxray14
    python scripts/medical_smoke.py            # runs all MedMNIST datasets
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch

from config import get_config
from data import MEDMNIST_REGISTRY, build_federated_datasets
from federated import Client, Server
from main import apply_dataset_defaults
from models import build_model
from utils import set_seed


def run(dataset: str) -> None:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    cfg = get_config(
        dataset=dataset,
        model="resnet18",
        pretrained=True,
        image_size=28,          # tiny + fast for the smoke test
        num_clients=5,
        alpha=0.5,
        num_rounds=2,
        local_epochs=1,
        candidate_size_d=4,
        active_size_m=2,
        batch_size=32,
        num_workers=0,
        device=device,
        exp_name=f"{dataset}_smoke",
    )
    cfg = apply_dataset_defaults(cfg)
    set_seed(cfg.seed)
    print(f"\n=== {dataset} (task={cfg.task}, C={cfg.num_classes}, dev={cfg.device}) ===")

    subsets, test_set, labels = build_federated_datasets(cfg)
    print(f"[smoke] sizes={[len(s) for s in subsets]} | test={len(test_set)}")
    clients = [Client(i, subsets[i], labels[i], cfg) for i in range(cfg.num_clients)]
    server = Server(cfg, clients, test_set, lambda: build_model(cfg))
    for r in range(cfg.num_rounds):
        info = server.run_round(r)
        print(f"round {r}: active={info['active']} "
              f"weights={[round(w, 3) for w in info['weights']]}")
    m = server.evaluate()
    headline = m.get("auc", m["acc"])
    key = "auc" if "auc" in m else "acc"
    print(f"eval loss={m['loss']:.3f} {key}={headline:.3f}")
    assert 0.0 <= m["acc"] <= 1.0
    print(f"{dataset} smoke OK")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(MEDMNIST_REGISTRY.keys())
    for d in targets:
        run(d)

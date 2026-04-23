"""Tiny end-to-end run to verify the pipeline works.

Uses MNIST (small + fast), 5 clients, 2 rounds. No network except for the one
MNIST download, which torchvision caches under ./data_cache.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from config import get_config
from data import build_federated_datasets
from federated import Client, Server
from models import build_model
from utils import set_seed


def run():
    cfg = get_config(
        dataset="mnist",
        num_clients=5,
        alpha=0.5,
        num_rounds=2,
        local_epochs=1,
        candidate_size_d=4,
        active_size_m=2,
        batch_size=64,
        device="cpu",
        exp_name="smoke",
    )
    set_seed(cfg.seed)
    subsets, test_set, labels = build_federated_datasets(cfg)
    clients = [Client(i, subsets[i], labels[i], cfg) for i in range(cfg.num_clients)]
    server = Server(cfg, clients, test_set, lambda: build_model(cfg))
    for r in range(cfg.num_rounds):
        info = server.run_round(r)
        print(
            f"round {r}: active={info['active']} "
            f"weights={[round(w, 3) for w in info['weights']]}"
        )
    m = server.evaluate()
    print(f"eval acc={m['acc']:.3f} loss={m['loss']:.3f}")
    assert 0.0 <= m["acc"] <= 1.0
    print("smoke OK")


if __name__ == "__main__":
    run()

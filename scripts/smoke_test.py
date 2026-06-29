"""Tiny end-to-end run to verify every method works.

Uses MNIST (small + fast), 6 clients, 2 rounds, on CPU. Runs all 9 methods
(FedAvg, FedProx, SCAFFOLD, FedDyn, FedBN, MOON, PoC-FedAvg, FedCE, Proposed)
and checks: no crash, aggregation weights sum to 1, Jain index in (0, 1].
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from config import METHOD_PRESETS, get_config
from data import build_federated_datasets
from federated import Client, Server
from models import build_model
from utils import set_seed


def run_one(method: str) -> None:
    cfg = get_config(
        method=method,
        dataset="mnist",
        num_clients=6,
        alpha=0.5,
        num_rounds=2,
        local_epochs=1,
        candidate_size_d=4,
        active_size_m=3,
        batch_size=64,
        device="cpu",
        local_test_frac=0.3,
        exp_name=f"smoke_{method}",
    )
    set_seed(cfg.seed)
    subsets, test_set, labels, test_subsets = build_federated_datasets(cfg)
    clients = [
        Client(i, subsets[i], labels[i], cfg, test_subset=test_subsets[i])
        for i in range(cfg.num_clients)
    ]
    server = Server(cfg, clients, test_set, lambda: build_model(cfg))
    for r in range(cfg.num_rounds):
        info = server.run_round(r)
        wsum = sum(info["weights"])
        assert abs(wsum - 1.0) < 1e-4, f"{method}: weights sum {wsum} != 1"
    m = server.evaluate()
    f = server.evaluate_per_client()
    assert 0.0 <= m["acc"] <= 1.0
    assert 0.0 < f["jain"] <= 1.0 + 1e-9, f"{method}: jain {f['jain']} out of range"
    print(
        f"  {method:11s} acc={m['acc']:.3f} loss={m['loss']:.3f} "
        f"jain={f['jain']:.3f} client_min={f['client_acc_min']:.3f}"
    )


def run() -> None:
    print("Smoke-testing all methods on MNIST (6 clients, 2 rounds, CPU):")
    for method in METHOD_PRESETS:
        run_one(method)
    print("smoke OK — all methods ran")


if __name__ == "__main__":
    run()

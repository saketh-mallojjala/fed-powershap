"""Entry point: ShapFed-WA aggregation + Power-of-Choice selection.

Usage:
    python main.py                       # full run with defaults (CIFAR-10)
    python main.py --dataset mnist --num-rounds 30 --active-size-m 3
    python main.py --selection-strategy random  # ablation
    python main.py --aggregation fedavg         # ablation
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, fields

import torch
from tqdm import tqdm

from config import Config, get_config
from data import build_federated_datasets
from federated import Client, Server
from models import build_model
from utils import JsonLogger, set_seed


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Generate CLI flags from Config fields.
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        kwargs = {"default": None, "dest": f.name}
        if f.type == bool or isinstance(f.default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(f.default) if f.default is not None else str
        p.add_argument(flag, **kwargs)
    return p


def apply_overrides(cfg: Config, args) -> Config:
    for f in fields(Config):
        v = getattr(args, f.name)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = apply_overrides(get_config(), args)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[cfg] device={cfg.device} dataset={cfg.dataset} N={cfg.num_clients} "
          f"m={cfg.active_size_m} d={cfg.candidate_size_d} "
          f"selection={cfg.selection_strategy} agg={cfg.aggregation}")

    set_seed(cfg.seed)

    # Data + clients.
    subsets, test_set, client_labels = build_federated_datasets(cfg)
    clients = [
        Client(i, subsets[i], client_labels[i], cfg)
        for i in range(cfg.num_clients)
    ]
    sizes = [c.size for c in clients]
    print(f"[data] per-client size min/median/max = "
          f"{min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}")

    server = Server(cfg, clients, test_set, lambda: build_model(cfg))

    with JsonLogger(cfg.log_dir, cfg.exp_name) as logger:
        logger.log({"type": "config", **asdict(cfg)})
        logger.log({
            "type": "client_sizes",
            "sizes": sizes,
            "class_histograms": [c.class_histogram.tolist() for c in clients],
        })

        pbar = tqdm(range(cfg.num_rounds), desc="rounds")
        best_acc = 0.0
        for r in pbar:
            t0 = time.time()
            round_info = server.run_round(r)
            round_info["wall"] = time.time() - t0
            logger.log({"type": "round", **round_info})

            if (r + 1) % cfg.eval_every == 0 or r == cfg.num_rounds - 1:
                metrics = server.evaluate()
                best_acc = max(best_acc, metrics["acc"])
                logger.log({"type": "eval", "round": r, **metrics})
                postfix = {
                    "acc": f"{metrics['acc']:.3f}",
                    "best": f"{best_acc:.3f}",
                    "loss": f"{metrics['loss']:.3f}",
                }
                if "qwk" in metrics:
                    postfix["qwk"] = f"{metrics['qwk']:.3f}"
                pbar.set_postfix(postfix)

        print(f"[done] best acc = {best_acc:.4f}; log: {logger.path}")


if __name__ == "__main__":
    main()

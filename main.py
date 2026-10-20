"""Entry point: ShapFed-WA aggregation + Power-of-Choice selection.

Usage:
    python main.py                       # full run with defaults (CIFAR-10)
    python main.py --dataset mnist --num-rounds 30 --active-size-m 3
    python main.py --selection-strategy random  # ablation
    python main.py --aggregation fedavg         # ablation
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, fields

import torch
from tqdm import tqdm

from config import Config, apply_method_preset, get_config
from data import MEDMNIST_REGISTRY, TABULAR_REGISTRY, build_federated_datasets
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


def apply_dataset_defaults(cfg: Config) -> Config:
    """Fill task / num_classes from the dataset registry so the user need not.

    For MedMNIST-backed datasets the task and class count are intrinsic, so we
    set them authoritatively. BraTS is a 2-class single-label problem.
    """
    if cfg.dataset in MEDMNIST_REGISTRY:
        entry = MEDMNIST_REGISTRY[cfg.dataset]
        cfg.num_classes = entry["num_classes"]
        cfg.task = entry["task"]
    elif cfg.dataset == "brats":
        cfg.num_classes = 2
        cfg.task = "single_label"
    elif cfg.dataset == "aptos":
        cfg.num_classes = 5  # DR grades 0-4 (ordinal); QWK is the headline metric
        cfg.task = "single_label"
    elif cfg.dataset in TABULAR_REGISTRY:
        # Binary screening task on a dense feature vector: force the MLP and read
        # the input width off the encoded data so --num-features never has to be
        # passed by hand (it depends on the feature regime and one-hot widths).
        from data.tabular import load_tabular
        _, _, meta = load_tabular(
            cfg.dataset, root=cfg.data_root, seed=cfg.seed,
            feature_regime=cfg.feature_regime,
        )
        cfg.num_classes = meta.num_classes
        cfg.task = "single_label"
        cfg.num_features = meta.num_features
        cfg.model = "mlp"
        print(f"[data] {cfg.dataset} regime={meta.feature_regime} "
              f"features={meta.num_features} train={meta.n_train} test={meta.n_test} "
              f"pos_rate={meta.positive_rate:.4f} dropped_rows={meta.dropped_rows}")
    return cfg


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = get_config()
    # Apply the method preset first (CLI default or chosen), then let any
    # explicit CLI flags override individual knobs (for ablations).
    cfg = apply_method_preset(cfg, args.method or cfg.method)
    cfg = apply_overrides(cfg, args)
    cfg = apply_dataset_defaults(cfg)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[cfg] device={cfg.device} dataset={cfg.dataset} N={cfg.num_clients} "
          f"method={cfg.method} m={cfg.active_size_m} d={cfg.candidate_size_d} "
          f"selection={cfg.selection_strategy} agg={cfg.aggregation} "
          f"solver={cfg.local_solver}")

    set_seed(cfg.seed)

    # Data + clients (per-client local test split drives the fairness metric).
    subsets, test_set, client_labels, test_subsets = build_federated_datasets(cfg)
    clients = [
        Client(i, subsets[i], client_labels[i], cfg, test_subset=test_subsets[i])
        for i in range(cfg.num_clients)
    ]
    # Tag the adversarial clients chosen by build_federated_datasets. With
    # --freerider-mode mixed each one gets a different attack type.
    for cid, mode in getattr(cfg, "freerider_assignment", {}).items():
        clients[cid].freerider_mode = mode

    # Leave-one-out: drop one client AFTER partitioning + noise injection, so the
    # remaining clients' data is bit-identical to the full federation. Selection
    # and all server state are positional over `clients`, so re-indexing is safe.
    excluded = int(getattr(cfg, "exclude_client", -1))
    if excluded >= 0:
        if not any(c.id == excluded for c in clients):
            raise ValueError(f"--exclude-client {excluded} not in 0..{len(clients)-1}")
        noisy_before = list(getattr(cfg, "noisy_client_ids", []))
        was_noisy = excluded in set(noisy_before)
        kept = [c.id for c in clients if c.id != excluded]
        clients = [c for c in clients if c.id != excluded]
        # Re-index positionally and remap the corrupted-client ground truth onto
        # the new indices, so the log stays self-consistent after exclusion.
        remap = {old: new for new, old in enumerate(kept)}
        for pos, c in enumerate(clients):
            c.id = pos
        cfg.noisy_client_ids = sorted(remap[i] for i in noisy_before if i != excluded)
        cfg.num_clients = len(clients)
        print(f"[loo] excluded client {excluded} (noisy={was_noisy}); "
              f"{cfg.num_clients} clients remain")

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
            # Ground truth for the mechanism analysis: which clients had their
            # labels corrupted. Set by build_federated_datasets; [] when clean.
            "noisy_client_ids": getattr(cfg, "noisy_client_ids", []),
            # Ground truth for reviewer A's question: a real covariate shift but
            # CORRECT labels. A loss-based score that down-weights these is
            # confusing "different" with "corrupted".
            "shifted_client_ids": getattr(cfg, "shifted_client_ids", []),
            # -1 unless this is a leave-one-out run; ids above are post-exclusion.
            "excluded_client": int(getattr(cfg, "exclude_client", -1)),
            # Ground truth for the C2 free-rider analysis.
            "freerider_client_ids": getattr(cfg, "freerider_client_ids", []),
            "freerider_mode": getattr(cfg, "freerider_mode", "none"),
            # {client_id: attack_name} -- lets the analysis separate outcomes
            # per attack type even when several run in the same federation.
            "freerider_assignment": {
                int(k): v for k, v in getattr(cfg, "freerider_assignment", {}).items()},
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
                fairness = server.evaluate_per_client()  # {} if no local test sets
                metrics.update(fairness)
                best_acc = max(best_acc, metrics["acc"])
                logger.log({"type": "eval", "round": r, **metrics})
                postfix = {
                    "acc": f"{metrics['acc']:.3f}",
                    "best": f"{best_acc:.3f}",
                    "loss": f"{metrics['loss']:.3f}",
                }
                if "qwk" in metrics:
                    postfix["qwk"] = f"{metrics['qwk']:.3f}"
                if "auc" in metrics:
                    postfix["auc"] = f"{metrics['auc']:.3f}"
                if "jain" in metrics:
                    postfix["jain"] = f"{metrics['jain']:.3f}"
                pbar.set_postfix(postfix)

        print(f"[done] best acc = {best_acc:.4f}; log: {logger.path}")


if __name__ == "__main__":
    main()
    # The JSONL log is flushed/closed by the JsonLogger context manager above,
    # so a hard exit here is safe. It sidesteps occasional hangs in DataLoader
    # worker / MPS teardown that would otherwise stall a multi-run script.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

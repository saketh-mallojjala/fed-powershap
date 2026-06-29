"""Full baseline-matrix runner: method x dataset x seed.

Spawns one ``main.py`` process per (method, dataset, seed) with consistent
config and log naming (``{dataset}_{method}_seed{n}-*.jsonl``). Skips runs whose
log already exists so it is resumable. Use ``--dry-run`` to print the plan.

Examples:
    # Fast iteration: proposed vs fedavg on octmnist, 1 seed, few rounds.
    python scripts/run_experiments.py --datasets octmnist \
        --methods proposed fedavg --seeds 0 --rounds 15 --image-size 64

    # Full paper sweep (9 methods x 4 datasets x 3 seeds).
    python scripts/run_experiments.py
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import METHOD_PRESETS  # noqa: E402

ALL_METHODS = list(METHOD_PRESETS.keys())
ALL_DATASETS = ["aptos", "octmnist", "isic", "chestxray14"]

# Per-dataset config knobs (medical FL protocol). Class count / task come from
# the dataset registry automatically (apply_dataset_defaults in main.py).
DATASET_CFG = {
    "aptos":       dict(model="resnet18", local_lr=0.001),
    "octmnist":    dict(model="resnet18", local_lr=0.001),
    "isic":        dict(model="resnet18", local_lr=0.001),
    "chestxray14": dict(model="resnet18", local_lr=0.001),
}


def build_cmd(py, method, dataset, seed, args) -> list:
    dcfg = DATASET_CFG.get(dataset, {})
    exp = f"{dataset}_{method}_seed{seed}"
    cmd = [
        py, os.path.join(ROOT, "main.py"),
        "--method", method,
        "--dataset", dataset,
        "--seed", str(seed),
        "--exp-name", exp,
        "--model", dcfg.get("model", "resnet18"),
        "--local-lr", str(dcfg.get("local_lr", 0.001)),
        "--image-size", str(args.image_size),
        "--num-clients", str(args.num_clients),
        "--alpha", str(args.alpha),
        "--num-rounds", str(args.rounds),
        "--candidate-size-d", str(args.candidate_d),
        "--active-size-m", str(args.active_m),
        "--local-epochs", str(args.local_epochs),
        "--batch-size", str(args.batch_size),
        "--local-test-frac", str(args.local_test_frac),
        "--eval-every", str(args.eval_every),
        "--noisy-client-frac", str(args.noisy_client_frac),
        "--label-noise-rate", str(args.label_noise_rate),
        "--num-workers", str(args.num_workers),
    ]
    if args.max_train:
        cmd += ["--max-train-samples", str(args.max_train)]
    return cmd, exp


def already_done(exp: str, log_dir: str) -> bool:
    return len(glob.glob(os.path.join(log_dir, f"{exp}-*.jsonl"))) > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--candidate-d", type=int, default=10)
    p.add_argument("--active-m", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--local-test-frac", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--noisy-client-frac", type=float, default=0.0)
    p.add_argument("--label-noise-rate", type=float, default=0.0)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    p.add_argument("--force", action="store_true", help="re-run even if a log exists")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    py = sys.executable or "python3"
    runs = [
        (m, d, s)
        for d in args.datasets
        for m in args.methods
        for s in args.seeds
    ]
    print(f"[plan] {len(runs)} runs: {len(args.methods)} methods x "
          f"{len(args.datasets)} datasets x {len(args.seeds)} seeds, "
          f"{args.rounds} rounds each")

    done, ran, failed = 0, 0, []
    for i, (m, d, s) in enumerate(runs, 1):
        cmd, exp = build_cmd(py, m, d, s, args)
        if not args.force and already_done(exp, args.log_dir):
            done += 1
            print(f"[{i}/{len(runs)}] skip (exists): {exp}")
            continue
        print(f"[{i}/{len(runs)}] run: {exp}")
        if args.dry_run:
            print("   " + " ".join(cmd))
            continue
        t0 = time.time()
        rc = subprocess.call(cmd)
        dt = time.time() - t0
        if rc == 0:
            ran += 1
            print(f"   ok ({dt:.0f}s)")
        else:
            failed.append(exp)
            print(f"   FAILED rc={rc} ({dt:.0f}s)")

    print(f"\n[summary] ran={ran} skipped={done} failed={len(failed)}")
    if failed:
        print("  failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()

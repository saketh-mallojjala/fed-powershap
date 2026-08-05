#!/usr/bin/env python3
"""Sweep driver for the tabular mental-health study (mirrors run_noise_sweep.py).

One FROZEN protocol for every method, so the comparison is apples-to-apples. The
local LR was calibrated on FedAvg alone (clean, seed 0) and then fixed for all
methods; no per-method tuning.

Logs are named ``{dataset}_{regime}_{method}_n{NN}_seed{s}``. Resumable: a run
whose log already exists is skipped unless --force.

Usage:
    python3 scripts/run_tabular_sweep.py
    python3 scripts/run_tabular_sweep.py --noises 0.0 0.4 --seeds 0 1 2
    python3 scripts/run_tabular_sweep.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Frozen protocol. Mirrors scripts/run_noise_sweep.py (N=10, m=5, 2 local epochs,
# 40 rounds, cosine LR, alpha=0.3, 20% local test) with the image-specific knobs
# replaced by the tabular MLP. local_lr=0.005 from the FedAvg-only calibration.
PROTOCOL = dict(
    num_clients=10, candidate_size_d=10, active_size_m=5, num_rounds=40,
    local_epochs=2, local_lr=0.005, batch_size=128, alpha=0.3,
    local_test_frac=0.2, eval_every=2, lr_schedule="cosine", num_workers=0,
    hidden_dim=64, mlp_depth=2, mlp_dropout=0.1,
)

LABEL_NOISE_RATE = 0.6   # 60% of a corrupted client's labels flipped, as before


def log_exists(log_dir, exp):
    return bool(glob.glob(os.path.join(log_dir, f"{exp}-*.jsonl")))


def build_cmd(method, dataset, regime, noise, seed, log_dir, extra=None):
    exp = f"{dataset}_{regime}_{method}_n{round(noise * 100):02d}_seed{seed}"
    cmd = [sys.executable, os.path.join(ROOT, "main.py"),
           "--method", method, "--dataset", dataset, "--seed", str(seed),
           "--exp-name", exp, "--device", "cpu", "--log-dir", log_dir,
           "--feature-regime", regime,
           "--noisy-client-frac", str(noise),
           "--label-noise-rate", str(LABEL_NOISE_RATE)]
    for k, v in PROTOCOL.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    if extra:
        cmd += extra
    return exp, cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="student_depression")
    p.add_argument("--regimes", nargs="+", default=["screening"])
    p.add_argument("--methods", nargs="+",
                   default=["fedavg", "fedprox", "feddyn", "fedce", "proposed"])
    p.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--log-dir",
                   default=os.path.join(ROOT, "studies", "tabular_mentalhealth", "logs"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    jobs = [(m, r, n, s)
            for r in args.regimes for n in args.noises
            for m in args.methods for s in args.seeds]
    print(f"[sweep] {len(jobs)} runs planned -> {args.log_dir}")

    done = skipped = failed = 0
    for i, (method, regime, noise, seed) in enumerate(jobs, 1):
        exp, cmd = build_cmd(method, args.dataset, regime, noise, seed, args.log_dir)
        if not args.force and log_exists(args.log_dir, exp):
            skipped += 1
            continue
        if args.dry_run:
            print(" ", " ".join(cmd))
            continue
        print(f"[{i}/{len(jobs)}] {exp}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            failed += 1
            print(f"  FAILED (rc={r.returncode})\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        else:
            done += 1

    print(f"[sweep] done={done} skipped={skipped} failed={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

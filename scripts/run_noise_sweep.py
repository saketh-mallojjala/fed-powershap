"""Noise-sweep driver for the robustness story (Workstream B).

Runs {methods} x {noise levels} x {seeds} on a single FROZEN protocol so the
comparison is apples-to-apples (fixed N, rounds, LR schedule, data cap). Logs
are named ``{dataset}_{method}_n{NN}_seed{s}`` where NN = round(noise*100), so
each noise level is a distinct run and nothing collides. Resumable: a run whose
log already exists is skipped unless --force.

Usage:
    python3 scripts/run_noise_sweep.py                      # default OCTMNIST batch
    python3 scripts/run_noise_sweep.py --noises 0.0 0.4 --seeds 0 1 2
    python3 scripts/run_noise_sweep.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Frozen protocol shared by every run in the sweep.
PROTOCOL = dict(
    model="resnet18", image_size=64, num_clients=10, candidate_size_d=10,
    active_size_m=5, num_rounds=40, local_epochs=2, local_lr=0.001,
    batch_size=128, max_train_samples=20000, alpha=0.3, local_test_frac=0.2,
    eval_every=2, lr_schedule="cosine", num_workers=0,
)


def log_exists(log_dir, exp):
    return bool(glob.glob(os.path.join(log_dir, f"{exp}-*.jsonl")))


def build_cmd(method, dataset, noise, seed, log_dir):
    exp = f"{dataset}_{method}_n{round(noise * 100):02d}_seed{seed}"
    cmd = [sys.executable, os.path.join(ROOT, "main.py"),
           "--method", method, "--dataset", dataset, "--seed", str(seed),
           "--exp-name", exp, "--device", "mps", "--log-dir", log_dir,
           "--noisy-client-frac", str(noise), "--label-noise-rate", "0.6"]
    for k, v in PROTOCOL.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return exp, cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="octmnist")
    p.add_argument("--methods", nargs="+", default=["fedavg", "feddyn", "proposed"])
    p.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.4])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    plan = [(m, args.dataset, n, s)
            for n in args.noises for s in args.seeds for m in args.methods]
    print(f"[sweep] {len(plan)} runs: methods={args.methods} "
          f"noises={args.noises} seeds={args.seeds} dataset={args.dataset}")

    done = ran = 0
    for i, (method, dataset, noise, seed) in enumerate(plan, 1):
        exp, cmd = build_cmd(method, dataset, noise, seed, args.log_dir)
        if not args.force and log_exists(args.log_dir, exp):
            print(f"[skip {i}/{len(plan)}] {exp} (log exists)")
            done += 1
            continue
        print(f"[run  {i}/{len(plan)}] {exp}", flush=True)
        if args.dry_run:
            print("   " + " ".join(cmd))
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[warn] {exp} exited rc={rc}")
        else:
            ran += 1
    print(f"[sweep] finished: {ran} ran, {done} skipped")


if __name__ == "__main__":
    main()

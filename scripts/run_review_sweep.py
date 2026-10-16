#!/usr/bin/env python3
"""Revision experiments for the conference reviews.

Every run is one `main.py` invocation under a frozen protocol; only the swept
axis changes. Results are appended to a single JSONL as each run finishes, so
a crash or an interrupt never loses completed work.

Axes, and which reviewer point each answers:

  noise      corruption level 0.0 .. 0.6              A.3, B.2
  type       symmetric vs asymmetric (pair-flip)      B.2
  shift      covariate shift, labels CORRECT          A.2
  alpha      Dirichlet 0.1 / 0.3 / 1.0                A.3
  part       active clients per round 2 / 5 / 8       A.3
  scale      10 / 20 / 50 clients                     B.5
  poison     model-poisoning adversaries              B.3

Usage:
  python3 scripts/run_review_sweep.py --dataset aptos --tier 1
  python3 scripts/run_review_sweep.py --dataset aptos --tier 1 --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "studies" / "review"
OUT.mkdir(parents=True, exist_ok=True)

# Frozen protocol from Table I of the submission.
BASE = dict(model="resnet18", image_size=64, local_epochs=2, local_lr=0.001,
            num_rounds=40, batch_size=128, num_workers=0, pretrained=True,
            partition_mode="dirichlet", lr_schedule="cosine", eval_every=40)

METHODS = ["proposed", "feddyn", "fednoro", "fedcorr", "feda3i"]
# Subsample caps keep the sweep tractable; 0 means use everything. OCTMNIST has
# 97,477 training images and costs ~6.3 h per 40-round run at full size, which
# no sweep can absorb. The cap is disclosed in the revision.
MAX_TRAIN = {"octmnist": 10000, "isic": 0, "aptos": 0}


def runs_for(dataset: str, tier: int, seeds: list[int]) -> list[dict]:
    """Each entry is {tag, axis, overrides}. `tag` must identify the run."""
    out = []

    def add(axis, method, seed, **ov):
        bits = [f"{k}{v}" for k, v in sorted(ov.items())]
        out.append(dict(axis=axis, method=method, seed=seed, overrides=ov,
                        tag=f"{dataset}_{axis}_{method}_s{seed}_" + "_".join(bits)))

    if tier == 1:
        # Corruption level x method. The core grid: reviewer A.3 and B.2.
        for seed in seeds:
            for nf in (0.0, 0.2, 0.4, 0.6):
                for m in METHODS:
                    ov = dict(num_clients=10, active_size_m=5, candidate_size_d=10,
                              alpha=0.3, noisy_client_frac=nf,
                              label_noise_rate=0.0 if nf == 0 else 0.6,
                              label_noise_type="symmetric")
                    add("noise", m, seed, **ov)
    elif tier == 2:
        s = seeds[0]
        # asymmetric noise at the submitted operating point (B.2)
        for m in METHODS:
            add("type", m, s, num_clients=10, active_size_m=5, candidate_size_d=10,
                alpha=0.3, noisy_client_frac=0.4, label_noise_rate=0.6,
                label_noise_type="asymmetric")
        # covariate shift with CORRECT labels (A.2)
        for m in METHODS:
            add("shift", m, s, num_clients=10, active_size_m=5, candidate_size_d=10,
                alpha=0.3, shift_client_frac=0.4, shift_severity=0.8,
                noisy_client_frac=0.0, label_noise_rate=0.0)
        # mixed: corrupted AND shifted clients present together (A.2, the hard case)
        for m in ("proposed", "feddyn"):
            add("mixed", m, s, num_clients=10, active_size_m=5, candidate_size_d=10,
                alpha=0.3, noisy_client_frac=0.3, label_noise_rate=0.6,
                shift_client_frac=0.3, shift_severity=0.8)
        # model poisoning (B.3)
        for atk in ("random_update", "scaled_global", "norm_inflation"):
            for m in ("proposed", "feddyn"):
                add("poison", m, s, num_clients=10, active_size_m=5,
                    candidate_size_d=10, alpha=0.3, freerider_frac=0.3,
                    freerider_mode=atk)
    elif tier == 3:
        s = seeds[0]
        # non-IID degree and participation ratio (A.3)
        for m in ("proposed", "feddyn"):
            for a in (0.1, 1.0):
                add("alpha", m, s, num_clients=10, active_size_m=5,
                    candidate_size_d=10, alpha=a, noisy_client_frac=0.4,
                    label_noise_rate=0.6)
            for am in (2, 8):
                add("part", m, s, num_clients=10, active_size_m=am,
                    candidate_size_d=10, alpha=0.3, noisy_client_frac=0.4,
                    label_noise_rate=0.6)
            # scaling (B.5)
            for nc in (20, 50):
                add("scale", m, s, num_clients=nc, active_size_m=max(2, nc // 2),
                    candidate_size_d=nc, alpha=0.3, noisy_client_frac=0.4,
                    label_noise_rate=0.6)
    else:
        raise SystemExit(f"unknown tier {tier}")
    return out


def already_done(path: Path, tag: str) -> bool:
    if not path.exists():
        return False
    with open(path) as f:
        for line in f:
            try:
                if json.loads(line).get("tag") == tag:
                    return True
            except Exception:
                pass
    return False


def read_result(log_dir: Path, exp: str) -> dict:
    """Pull the final eval and the mechanism ground truth out of the run log."""
    files = sorted(glob.glob(str(log_dir / f"{exp}-*.jsonl")))
    if not files:
        return {}
    last_eval, sizes, cfg = None, None, None
    weights_by_round = []
    with open(files[-1]) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("type")
            if t == "eval":
                last_eval = d
            elif t == "client_sizes":
                sizes = d
            elif t == "config":
                cfg = d
            elif t == "round" and d.get("weights") is not None:
                weights_by_round.append({"active": d.get("active"),
                                         "weights": d.get("weights")})
    res = {}
    if last_eval:
        res.update({k: v for k, v in last_eval.items()
                    if k not in ("type", "round")})
    if sizes:
        res["noisy_client_ids"] = sizes.get("noisy_client_ids", [])
        res["shifted_client_ids"] = sizes.get("shifted_client_ids", [])
        res["freerider_client_ids"] = sizes.get("freerider_client_ids", [])
        res["client_sizes"] = sizes.get("sizes")
    if cfg:
        res["num_clients"] = cfg.get("num_clients")
    res["weights_by_round"] = weights_by_round
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = runs_for(args.dataset, args.tier, args.seeds)
    results = OUT / f"{args.dataset}_tier{args.tier}.jsonl"
    log_dir = OUT / "logs"
    log_dir.mkdir(exist_ok=True)

    pending = [j for j in jobs if not already_done(results, j["tag"])]
    print(f"{args.dataset} tier {args.tier}: {len(jobs)} runs, "
          f"{len(jobs) - len(pending)} already done, {len(pending)} to run")
    if args.dry_run:
        for j in pending:
            print("  ", j["tag"])
        return

    for i, j in enumerate(pending, 1):
        ov = dict(BASE)
        ov.update(j["overrides"])
        mt = MAX_TRAIN.get(args.dataset, 0)
        if mt:
            ov["max_train_samples"] = mt
        cmd = [sys.executable, str(ROOT / "main.py"),
               "--dataset", args.dataset, "--method", j["method"],
               "--seed", str(j["seed"]), "--device", args.device,
               "--exp-name", j["tag"], "--log-dir", str(log_dir)]
        for k, v in ov.items():
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                cmd.append(flag if v else flag.replace("--", "--no-", 1))
            else:
                cmd += [flag, str(v)]
        t0 = time.time()
        print(f"[{i}/{len(pending)}] {j['tag']}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        if p.returncode != 0:
            print("   FAILED\n" + "\n".join(p.stdout.splitlines()[-6:]), flush=True)
            rec = dict(j, failed=True, wall=wall,
                       stderr="\n".join(p.stderr.splitlines()[-6:]))
        else:
            rec = dict(j, failed=False, wall=wall,
                       **read_result(log_dir, j["tag"]))
            print(f"   acc={rec.get('acc')} bacc={rec.get('bacc_opt', rec.get('acc'))} "
                  f"{wall/60:.1f} min", flush=True)
        with open(results, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    print("SWEEP_DONE")


if __name__ == "__main__":
    main()

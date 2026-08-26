#!/usr/bin/env python3
"""Two experiments: Dirichlet-alpha sensitivity, and the full baseline comparison.

A. ALPHA SWEEP. Dirichlet alpha controls how unevenly labels are spread across
   clients: small alpha = extreme label skew, large alpha ~ IID. Two questions:
     - does DCFI hold up as heterogeneity varies (robustness), and
     - does reward fidelity TRACK heterogeneity?
   The second matters because our natural-partition runs scored rho 0.87 while
   the synthetic Dirichlet ones scored 0.17. The hypothesis is that fidelity is
   limited by how different the clients actually are: if every client
   contributes the same, there is nothing for a reward rule to rank. The alpha
   sweep tests that directly, on one dataset with everything else fixed.

B. BASELINE COMPARISON. Every aggregator in the harness on one frozen protocol.
   Reward metrics apply only to DCFI (no baseline defines a reward), so the
   reward columns are compared against the two rules any system could use
   without a mechanism: equal-split and size-proportional.

Usage:
    python3 scripts/run_alpha_baselines.py --experiment alpha
    python3 scripts/run_alpha_baselines.py --experiment baselines
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STUDY = os.path.join(ROOT, "studies", "tabular_mentalhealth")
sys.path.insert(0, ROOT)

from scripts.run_dcfi import (  # noqa: E402
    COLLAPSE, convergence_round, read_log, reward_metrics,
)

PROTOCOL = dict(
    dataset="student_depression", feature_regime="screening",
    num_clients=10, candidate_size_d=10, active_size_m=5, num_rounds=40,
    local_epochs=2, local_lr=0.005, batch_size=128,
    local_test_frac=0.2, lr_schedule="cosine", num_workers=0,
    hidden_dim=64, mlp_depth=2, mlp_dropout=0.1,
)
ALPHAS = [0.1, 0.3, 0.5, 1.0, 10.0]      # 10.0 is effectively IID
BASELINES = ["fedavg", "fedprox", "scaffold", "feddyn", "fedbn", "moon", "fedce"]


def build_cmd(method, attack, seed, alpha, frac, exp, log_dir):
    cmd = [sys.executable, os.path.join(ROOT, "main.py"),
           "--method", method, "--seed", str(seed), "--exp-name", exp,
           "--device", "cpu", "--log-dir", log_dir, "--eval-every", "2",
           "--alpha", str(alpha)]
    if method == "dcfi":
        cmd += ["--dcfi-gate-mode", "reward_only"]
    if attack == "label_flip":
        cmd += ["--noisy-client-frac", str(frac), "--label-noise-rate", "0.6"]
    elif attack != "none":
        cmd += ["--freerider-mode", attack, "--freerider-frac", str(frac)]
    for k, v in PROTOCOL.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd


def run_jobs(jobs, log_dir):
    todo = [(e, c) for e, c in jobs
            if not glob.glob(os.path.join(log_dir, f"{e}-*.jsonl"))]
    print(f"[sweep] {len(jobs)} planned, {len(todo)} to run", flush=True)
    for i, (exp, cmd) in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {exp}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-1200:])
            raise SystemExit(f"failed: {exp}")


def _ms(v, p=3):
    return f"{np.mean(v):.{p}f}±{np.std(v):.{p}f}" if len(v) else "—"


def collect(log_dir, prefix, seeds, want_reward=False):
    perf, rew = defaultdict(list), defaultdict(list)
    for s in seeds:
        run = read_log(log_dir, f"{prefix}_seed{s}")
        if not run:
            continue
        last = run["last"]
        for k in ("balanced_acc", "auc", "f1_pos", "client_acc_min", "jain"):
            if k in last:
                perf[k].append(last[k])
        perf["collapsed"].append(1.0 if last.get("balanced_acc", 1) <= COLLAPSE else 0.0)
        cr = convergence_round(run["evals"])
        if cr is not None:
            perf["conv"].append(cr)
        if want_reward and run["rounds"] and run["rounds"][0].get("dcfi"):
            for k, v in reward_metrics(run, 0.05).items():
                if np.isfinite(v):
                    rew[k].append(v)
    return perf, rew


def experiment_alpha(args):
    log_dir = os.path.join(STUDY, "logs_alpha")
    os.makedirs(log_dir, exist_ok=True)
    jobs = []
    for a in ALPHAS:
        tag = str(a).replace(".", "p")
        for atk in args.attacks:
            for m in ["feddyn", "dcfi"]:
                for s in args.seeds:
                    exp = f"a{tag}_{m}_{atk}_seed{s}"
                    jobs.append((exp, build_cmd(m, atk, s, a, args.frac, exp, log_dir)))
    if not args.skip_runs:
        run_jobs(jobs, log_dir)

    L = ["# A. Dirichlet-alpha sensitivity\n",
         f"{len(args.seeds)} seeds, student_depression, everything except alpha held "
         "fixed. Small alpha = extreme label skew across clients; alpha=10 is "
         "effectively IID.\n",
         "## A1. Does the proposed system hold up as heterogeneity varies?\n",
         "| alpha | Attack | FedDyn bal-acc | DCFI bal-acc | Δ | FedDyn collapsed | DCFI collapsed |",
         "|---|---|---|---|---|---|---|"]
    for a in ALPHAS:
        tag = str(a).replace(".", "p")
        for atk in args.attacks:
            pf, _ = collect(log_dir, f"a{tag}_feddyn_{atk}", args.seeds)
            pd_, _ = collect(log_dir, f"a{tag}_dcfi_{atk}", args.seeds)
            if not pf["balanced_acc"] or not pd_["balanced_acc"]:
                continue
            d = np.mean(pd_["balanced_acc"]) - np.mean(pf["balanced_acc"])
            L.append(f"| {a} | {atk} | {_ms(pf['balanced_acc'])} | {_ms(pd_['balanced_acc'])} | "
                     f"{d:+.3f} | {int(sum(pf['collapsed']))}/{len(pf['collapsed'])} | "
                     f"{int(sum(pd_['collapsed']))}/{len(pd_['collapsed'])} |")

    L += ["\n## A2. Does reward fidelity track client heterogeneity?\n",
          "The hypothesis from the natural-partition runs: a reward rule can only "
          "rank clients that genuinely differ. Lower alpha => more different clients "
          "=> fidelity should rise. Reference points: synthetic alpha=0.3 gave "
          "rho 0.165, real hospital/specialty partitions gave 0.72-0.87.\n",
          "| alpha | ρ(reward, LOO) | Adversary reward share | IR | Jain(reward) | Jain(selection) |",
          "|---|---|---|---|---|---|"]
    for a in ALPHAS:
        tag = str(a).replace(".", "p")
        atk = args.attacks[0]
        _, rw = collect(log_dir, f"a{tag}_dcfi_{atk}", args.seeds, want_reward=True)
        if not rw:
            continue
        L.append(f"| {a} | {_ms(rw['reward_contrib_spearman'])} | "
                 f"{np.mean(rw['adversary_reward_share'])*100:.2f}% | "
                 f"{np.mean(rw['individual_rationality'])*100:.0f}% | "
                 f"{_ms(rw['jain_useful_rewards'])} | {_ms(rw['jain_selection'])} |")
    return L, os.path.join(STUDY, "tables", "alpha_sensitivity.md")


def experiment_baselines(args):
    log_dir = os.path.join(STUDY, "logs_baselines")
    os.makedirs(log_dir, exist_ok=True)
    methods = BASELINES + ["dcfi"]
    jobs = []
    for atk in args.attacks:
        for m in methods:
            for s in args.seeds:
                exp = f"b_{m}_{atk}_seed{s}"
                jobs.append((exp, build_cmd(m, atk, s, args.alpha, args.frac, exp, log_dir)))
    if not args.skip_runs:
        run_jobs(jobs, log_dir)

    L = ["# B. Baseline comparison\n",
         f"{len(args.seeds)} seeds, student_depression, Dirichlet alpha={args.alpha}, "
         f"adversary fraction {args.frac:.0%}. One frozen protocol; the local learning "
         "rate was calibrated on FedAvg alone and then fixed for every method, so no "
         "method is tuned in its own favour.\n"]
    for atk in args.attacks:
        L += [f"\n## {atk}\n",
              "| Method | Balanced acc | AUC | F1(pos) | Jain(acc) | Worst-client | Conv. | Collapsed |",
              "|---|---|---|---|---|---|---|---|"]
        rows = []
        for m in methods:
            pf, _ = collect(log_dir, f"b_{m}_{atk}", args.seeds)
            if not pf["balanced_acc"]:
                continue
            rows.append((np.mean(pf["balanced_acc"]), m, pf))
        for _, m, pf in sorted(rows, reverse=True):
            star = "**" if m == "dcfi" else ""
            L.append(f"| {star}{m}{star} | {_ms(pf['balanced_acc'])} | {_ms(pf['auc'])} | "
                     f"{_ms(pf['f1_pos'])} | {_ms(pf['jain'])} | {_ms(pf['client_acc_min'])} | "
                     f"{np.mean(pf['conv']):.0f} | {int(sum(pf['collapsed']))}/{len(pf['collapsed'])} |")

    L += ["\n## Incentive properties — only DCFI provides them\n",
          "No baseline aggregator defines a reward at all, so the comparison is "
          "against the two rules a system could use without a mechanism.\n",
          "| Rule | Adversary reward share | ρ(reward, LOO) | IR |",
          "|---|---|---|---|"]
    for atk in args.attacks:
        if atk == "none":
            continue
        _, rw = collect(log_dir, f"b_dcfi_{atk}", args.seeds, want_reward=True)
        if rw:
            L.append(f"| **DCFI** ({atk}) | {np.mean(rw['adversary_reward_share'])*100:.2f}% | "
                     f"{_ms(rw['reward_contrib_spearman'])} | "
                     f"{np.mean(rw['individual_rationality'])*100:.0f}% |")
    L.append(f"| equal-split (any baseline) | {args.frac*100:.0f}% (their full share) | 0.000 | n/a |")
    L.append("| size-proportional (FedAvg-style) | ~their full share | n/a | n/a |")
    return L, os.path.join(STUDY, "tables", "baseline_comparison.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["alpha", "baselines"], required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--attacks", nargs="+", default=["none", "zero_update"])
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--frac", type=float, default=0.4)
    p.add_argument("--skip-runs", action="store_true")
    args = p.parse_args()

    L, out = (experiment_alpha(args) if args.experiment == "alpha"
              else experiment_baselines(args))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

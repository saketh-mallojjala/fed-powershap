#!/usr/bin/env python3
"""Criterion C1: reward fidelity against a leave-one-out ground truth.

The pre-registered primary test in docs/INCENTIVE_PROPOSAL.md §10.1:

    Spearman rho(reward, LOO value) must exceed EACH of size-proportional,
    equal-split, and raw-CSSV rewards by >= 0.15, with non-overlapping 95%
    bootstrap CIs over 5 seeds.

Ground truth
------------
For client k, the true contribution is its leave-one-out marginal value

    LOO_k = V(N) - V(N \\ {k})

where V is the balanced accuracy of the trained global model on the shared,
clean test set at the last round. V is measured under **FedAvg**, deliberately:
using the proposed method's own weighting to define the ground truth would be
circular, since that weighting is exactly what we are scoring. FedAvg is
size-weighted with no quality signal, so it is mechanism-neutral.

Excluding a client happens after partitioning and noise injection (see
main.py --exclude-client), so every other client's data is bit-identical
between V(N) and V(N \\ {k}) -- that is what makes the difference a marginal
value rather than a re-randomisation.

Reward rules compared (all budget-balanced, B=1 per round, accumulated)
----------------------------------------------------------------------
  fedshapin  s_k = 0.5*q_k + 0.5*rep_k    (§8.1 with alpha=0, i.e. no CSSV term)
             q_k = normalized exp(-beta * (loss_k - min_loss)), the §5.2 detector
             rep_k = EMA of q_k, the reputation term
  quality    q_k alone (ablation: does the reputation term earn its place?)
  cssv       raw ShapFed CSSV score, from a --contrib-signal cssv run
  size       split proportional to |D_k|      (the FedAvg-style proxy)
  equal      split equally among active clients

Usage:
    python3 scripts/loo_reward_fidelity.py                 # full experiment
    python3 scripts/loo_reward_fidelity.py --seeds 0 1     # quick
    python3 scripts/loo_reward_fidelity.py --skip-runs     # analyse existing logs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STUDY = os.path.join(ROOT, "studies", "tabular_mentalhealth")

# Frozen protocol -- identical to the main sweep (scripts/run_tabular_sweep.py).
PROTOCOL = dict(
    dataset="student_depression", feature_regime="screening",
    num_clients=10, candidate_size_d=10, active_size_m=5, num_rounds=40,
    local_epochs=2, local_lr=0.005, batch_size=128, alpha=0.3,
    local_test_frac=0.2, lr_schedule="cosine", num_workers=0,
    hidden_dim=64, mlp_depth=2, mlp_dropout=0.1, label_noise_rate=0.6,
)
LOSS_BETA = 1.5      # matches METHOD_PRESETS["proposed"]["loss_weight_beta"]
EMA_MOM = 0.5        # matches cssv_ema


def build_cmd(method, seed, noise, exp, log_dir, exclude=-1, contrib_signal=None):
    cmd = [sys.executable, os.path.join(ROOT, "main.py"),
           "--method", method, "--seed", str(seed), "--exp-name", exp,
           "--device", "cpu", "--log-dir", log_dir,
           "--noisy-client-frac", str(noise),
           "--eval-every", "40"]          # last round is always evaluated
    if exclude >= 0:
        cmd += ["--exclude-client", str(exclude)]
    if contrib_signal:
        cmd += ["--contrib-signal", contrib_signal]
    for k, v in PROTOCOL.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd


def log_exists(log_dir, exp):
    return bool(glob.glob(os.path.join(log_dir, f"{exp}-*.jsonl")))


def read_log(log_dir, exp):
    paths = sorted(glob.glob(os.path.join(log_dir, f"{exp}-*.jsonl")))
    if not paths:
        return None
    evals, rounds, meta = [], [], {}
    with open(paths[-1]) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("type")
            if t == "eval":
                evals.append(rec)
            elif t == "round":
                rounds.append(rec)
            elif t == "client_sizes":
                meta = rec
    if not evals:
        return None
    return dict(last=evals[-1], rounds=rounds, meta=meta)


# ---------------- reward rules ----------------

def _normalize(x):
    x = np.asarray(x, dtype=float)
    s = x.sum()
    return x / s if s > 1e-12 else np.ones_like(x) / len(x)


def accumulate_rewards(run, n_clients, sizes, budget=1.0):
    """Per-rule cumulative reward vectors of length n_clients.

    Each round, budget B is split among that round's ACTIVE clients according
    to the rule, so every rule is strongly budget-balanced by construction and
    the comparison isolates the *signal*, not the normalisation.
    """
    out = {k: np.zeros(n_clients) for k in ("fedshapin", "quality", "size", "equal")}
    rep = np.zeros(n_clients)         # reputation EMA state
    for rd in run["rounds"]:
        active = rd.get("active") or []
        losses = rd.get("contrib_losses")
        if not active:
            continue
        a = np.array(active, dtype=int)

        out["equal"][a] += budget / len(a)
        out["size"][a] += budget * _normalize(sizes[a])

        if losses and len(losses) == len(active):
            lo = np.asarray(losses, dtype=float)
            q = _normalize(np.exp(-LOSS_BETA * (lo - lo.min())))
            rep[a] = EMA_MOM * rep[a] + (1 - EMA_MOM) * q
            out["quality"][a] += budget * q
            s = _normalize(0.5 * q + 0.5 * _normalize(rep[a]))
            out["fedshapin"][a] += budget * s
    return out


def accumulate_cssv_reward(run, n_clients, budget=1.0):
    """Raw-CSSV reward, from a --contrib-signal cssv run.

    Score per client = sum over classes of (cos+1)/2, i.e. exactly the ShapFed
    per-client contribution before any size-prior blending.
    """
    out = np.zeros(n_clients)
    got = False
    for rd in run["rounds"]:
        active, cssv = rd.get("active") or [], rd.get("cssv")
        if not active or not cssv:
            continue
        c = np.asarray(cssv, dtype=float)
        if c.ndim != 2 or c.shape[0] != len(active):
            continue
        score = ((c + 1.0) / 2.0).sum(axis=1)
        out[np.array(active, dtype=int)] += budget * _normalize(score)
        got = True
    return out if got else None


# ---------------- statistics ----------------

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        i = 0
        sv = v[order]
        while i < len(v):                      # average ranks within ties
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def bootstrap_ci(vals, n_boot=10000, seed=0, alpha=0.05):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.4])
    p.add_argument("--log-dir", default=os.path.join(STUDY, "logs_loo"))
    p.add_argument("--out", default=os.path.join(STUDY, "tables", "loo_reward_fidelity.md"))
    p.add_argument("--json-out", default=os.path.join(STUDY, "tables", "loo_reward_fidelity.json"))
    p.add_argument("--skip-runs", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    N = PROTOCOL["num_clients"]

    # ---------- 1. run everything ----------
    jobs = []
    for noise in args.noises:
        nn = f"{round(noise * 100):02d}"
        for seed in args.seeds:
            # ground truth: full federation + N leave-one-out federations
            jobs.append((f"loo_fedavg_n{nn}_seed{seed}_full",
                         build_cmd("fedavg", seed, noise,
                                   f"loo_fedavg_n{nn}_seed{seed}_full", args.log_dir)))
            for k in range(N):
                jobs.append((f"loo_fedavg_n{nn}_seed{seed}_x{k}",
                             build_cmd("fedavg", seed, noise,
                                       f"loo_fedavg_n{nn}_seed{seed}_x{k}",
                                       args.log_dir, exclude=k)))
            # reward signals
            jobs.append((f"loo_signal_loss_n{nn}_seed{seed}",
                         build_cmd("proposed", seed, noise,
                                   f"loo_signal_loss_n{nn}_seed{seed}", args.log_dir)))
            jobs.append((f"loo_signal_cssv_n{nn}_seed{seed}",
                         build_cmd("proposed", seed, noise,
                                   f"loo_signal_cssv_n{nn}_seed{seed}",
                                   args.log_dir, contrib_signal="cssv")))

    if not args.skip_runs:
        todo = [(e, c) for e, c in jobs if args.force or not log_exists(args.log_dir, e)]
        print(f"[loo] {len(jobs)} runs planned, {len(jobs) - len(todo)} already present, "
              f"{len(todo)} to run")
        for i, (exp, cmd) in enumerate(todo, 1):
            print(f"  [{i}/{len(todo)}] {exp}", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:]); print(r.stderr[-2000:])
                raise SystemExit(f"run failed: {exp}")

    # ---------- 2. analyse ----------
    METRIC = "balanced_acc"
    RULES = ["fedshapin", "quality", "cssv", "size", "equal"]
    results, per_seed = {}, {}

    for noise in args.noises:
        nn = f"{round(noise * 100):02d}"
        rhos = {r: [] for r in RULES}
        detail = []
        for seed in args.seeds:
            full = read_log(args.log_dir, f"loo_fedavg_n{nn}_seed{seed}_full")
            if full is None:
                continue
            v_full = full["last"][METRIC]
            loo = np.full(N, np.nan)
            for k in range(N):
                r = read_log(args.log_dir, f"loo_fedavg_n{nn}_seed{seed}_x{k}")
                if r is not None:
                    loo[k] = v_full - r["last"][METRIC]
            if not np.isfinite(loo).all():
                continue

            sizes = np.array(full["meta"]["sizes"], dtype=float)
            noisy = set(full["meta"].get("noisy_client_ids", []))

            sig = read_log(args.log_dir, f"loo_signal_loss_n{nn}_seed{seed}")
            rewards = accumulate_rewards(sig, N, sizes)
            cssv_run = read_log(args.log_dir, f"loo_signal_cssv_n{nn}_seed{seed}")
            cssv_rw = accumulate_cssv_reward(cssv_run, N) if cssv_run else None
            if cssv_rw is not None:
                rewards["cssv"] = cssv_rw

            for rule in RULES:
                if rule in rewards:
                    rhos[rule].append(spearman(rewards[rule], loo))
            detail.append(dict(
                seed=seed, v_full=v_full, loo=loo.tolist(),
                noisy_clients=sorted(noisy), sizes=sizes.tolist(),
                loo_noisy_mean=float(np.mean([loo[i] for i in range(N) if i in noisy]))
                if noisy else None,
                loo_clean_mean=float(np.mean([loo[i] for i in range(N) if i not in noisy]))
                if noisy else None,
                rewards={k: v.tolist() for k, v in rewards.items()},
            ))
        results[noise] = {r: bootstrap_ci(rhos[r], seed=13) for r in RULES}
        results[noise]["_raw"] = {r: rhos[r] for r in RULES}
        per_seed[noise] = detail

    # ---------- 3. report ----------
    L = ["# C1 — reward fidelity vs a leave-one-out ground truth\n",
         f"Spearman ρ between cumulative reward and LOO marginal value "
         f"(balanced accuracy, N={N} clients), over {len(args.seeds)} seeds. "
         "Ground truth measured under FedAvg (mechanism-neutral). "
         "95% CI by bootstrap over seeds (10k resamples).\n",
         "**Pre-registered bar (§10.1 C1):** fedshapin must beat *each* of "
         "`size`, `equal`, `cssv` by ≥ 0.15 with non-overlapping CIs.\n"]

    for noise in args.noises:
        L.append(f"\n### {'Clean (0% noisy clients)' if noise == 0 else f'{int(noise*100)}% noisy clients'}\n")
        L.append("| Reward rule | Spearman ρ | 95% CI | Δ vs fedshapin |")
        L.append("|---|---|---|---|")
        base = results[noise]["fedshapin"][0]
        for rule in RULES:
            m, lo, hi = results[noise][rule]
            if not np.isfinite(m):
                continue
            d = "—" if rule == "fedshapin" else f"{base - m:+.3f}"
            L.append(f"| {'**'+rule+'**' if rule=='fedshapin' else rule} | "
                     f"{m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {d} |")

        margins = [base - results[noise][r][0] for r in ("size", "equal", "cssv")
                   if np.isfinite(results[noise][r][0])]
        ok = margins and min(margins) >= 0.15
        L.append(f"\n**C1 at this noise level: {'PASS' if ok else 'FAIL'}** — "
                 f"smallest margin over a baseline is {min(margins):+.3f} "
                 f"(bar: +0.150).\n" if margins else "")

        # Does LOO value itself distinguish corrupted clients?
        d = per_seed[noise]
        if d and d[0].get("loo_noisy_mean") is not None:
            nz = np.mean([x["loo_noisy_mean"] for x in d])
            cl = np.mean([x["loo_clean_mean"] for x in d])
            L.append(f"Ground-truth sanity check: mean LOO value of corrupted "
                     f"clients = {nz:+.4f} vs clean = {cl:+.4f}.\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    json.dump({str(k): {"ci": {r: results[k][r] for r in RULES},
                        "raw": results[k]["_raw"], "per_seed": per_seed[k]}
               for k in results}, open(args.json_out, "w"), indent=2)
    print("\n".join(L))
    print(f"\nwrote {args.out}\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

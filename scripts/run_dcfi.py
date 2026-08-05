#!/usr/bin/env python3
"""DCFI-UPA evaluation: every attack in the taxonomy, DCFI vs baselines.

Runs {methods} x {attacks} x {seeds} on the frozen tabular protocol and reports
the three metric families the framework is specified against:

  Performance    balanced accuracy, macro-F1, AUC, convergence rounds,
                 collapsed seeds
  Reward fairness Spearman(reward, LOO contribution), reward-contribution MAE,
                 reward share of positive vs negative contributors,
                 reward-to-cost ratio, individual-rationality rate
  Fairness       Jain index of useful-client rewards, worst-client balanced
                 accuracy, selection-frequency Jain index, reward vs
                 client-size correlation, minority-class recall

Usage:
    python3 scripts/run_dcfi.py                 # full sweep then report
    python3 scripts/run_dcfi.py --skip-runs     # report from existing logs
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

from federated.dcfi import ATTACKS  # noqa: E402

PROTOCOL = dict(
    dataset="student_depression", feature_regime="screening",
    num_clients=10, candidate_size_d=10, active_size_m=5, num_rounds=40,
    local_epochs=2, local_lr=0.005, batch_size=128, alpha=0.3,
    local_test_frac=0.2, lr_schedule="cosine", num_workers=0,
    hidden_dim=64, mlp_depth=2, mlp_dropout=0.1,
)
METHODS = ["feddyn", "dcfi"]
COLLAPSE = 0.51        # balanced accuracy at/below this = single-class prediction


def build_cmd(method, attack, seed, frac, exp, log_dir):
    cmd = [sys.executable, os.path.join(ROOT, "main.py"),
           "--method", method, "--seed", str(seed), "--exp-name", exp,
           "--device", "cpu", "--log-dir", log_dir, "--eval-every", "2"]
    if attack == "label_flip":
        # Label flipping is a data corruption, not an update behaviour.
        cmd += ["--noisy-client-frac", str(frac), "--label-noise-rate", "0.6"]
    elif attack != "none":
        cmd += ["--freerider-mode", attack, "--freerider-frac", str(frac)]
    for k, v in PROTOCOL.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd


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
    return dict(evals=evals, last=evals[-1], rounds=rounds, meta=meta)


def convergence_round(evals, frac=0.95):
    """First round reaching `frac` of the run's own best balanced accuracy."""
    b = [e.get("balanced_acc", 0.0) for e in evals]
    if not b:
        return None
    target = frac * max(b)
    for e, v in zip(evals, b):
        if v >= target:
            return e.get("round")
    return evals[-1].get("round")


def jain(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or x.sum() <= 0:
        return float("nan")
    return float((x.sum() ** 2) / (len(x) * (x ** 2).sum()))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        sv = v[o]; i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    rx, ry = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def reward_metrics(run, cost_share):
    """Reward-fairness and fairness metrics for one DCFI run.

    Ground-truth contribution here is the within-round LOO utility DCFI already
    computes on its validation split, accumulated per client. It is an
    approximation to a full retraining LOO -- the framework's own 'periodic
    audit' quantity -- and is what the reward is scored against.
    """
    meta = run["meta"]
    n = len(meta["sizes"])
    sizes = np.array(meta["sizes"], float)
    adversaries = set(meta.get("freerider_client_ids", [])) | set(
        meta.get("noisy_client_ids", []))

    reward = np.zeros(n)
    contrib = np.zeros(n)
    selections = np.zeros(n)
    rounds_paid = np.zeros(n)
    for r in run["rounds"]:
        d = r.get("dcfi")
        if not d:
            continue
        for cid in r.get("active", []):
            selections[cid] += 1
        for cid, v in d.get("reward", {}).items():
            reward[int(cid)] += float(v)
            rounds_paid[int(cid)] += 1
        for cid, u in zip(d.get("kept_ids", []), d.get("utility", [])):
            contrib[int(cid)] += float(u)

    out = {}
    honest = [c for c in range(n) if c not in adversaries]
    tot = reward.sum()

    out["reward_contrib_spearman"] = spearman(reward, contrib)
    # MAE between reward share and contribution share, on positive contributors.
    pos = contrib > 0
    if pos.any() and tot > 0:
        rs = reward[pos] / tot
        cs = contrib[pos] / contrib[pos].sum()
        out["reward_contrib_mae"] = float(np.mean(np.abs(rs - cs)))
    out["reward_share_positive"] = float(reward[contrib > 0].sum() / tot) if tot else np.nan
    out["reward_share_negative"] = float(reward[contrib <= 0].sum() / tot) if tot else np.nan
    out["adversary_reward_share"] = (
        float(sum(reward[c] for c in adversaries) / tot) if tot and adversaries else 0.0)

    # Reward-to-cost and individual rationality, over honest clients that
    # actually participated. Cost accrues per selected round.
    cost = selections * cost_share
    with np.errstate(divide="ignore", invalid="ignore"):
        rtc = np.where(cost > 0, reward / cost, np.nan)
    hon_rtc = np.array([rtc[c] for c in honest if np.isfinite(rtc[c])])
    out["reward_to_cost"] = float(np.mean(hon_rtc)) if len(hon_rtc) else np.nan
    out["individual_rationality"] = (
        float(np.mean(hon_rtc >= 1.0)) if len(hon_rtc) else np.nan)

    out["jain_useful_rewards"] = jain([reward[c] for c in honest])
    out["jain_selection"] = jain(selections)
    out["reward_size_corr"] = (
        float(np.corrcoef(reward, sizes)[0, 1]) if np.std(reward) > 0 else np.nan)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--attacks", nargs="+", default=["none"] + ATTACKS)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--frac", type=float, default=0.4)
    p.add_argument("--log-dir", default=os.path.join(STUDY, "logs_dcfi"))
    p.add_argument("--out", default=os.path.join(STUDY, "tables", "dcfi_results.md"))
    p.add_argument("--skip-runs", action="store_true")
    args = p.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    jobs = []
    for atk in args.attacks:
        for meth in args.methods:
            for s in args.seeds:
                exp = f"dcfi_{meth}_{atk}_seed{s}"
                jobs.append((exp, build_cmd(meth, atk, s, args.frac, exp, args.log_dir)))

    if not args.skip_runs:
        todo = [(e, c) for e, c in jobs
                if not glob.glob(os.path.join(args.log_dir, f"{e}-*.jsonl"))]
        print(f"[dcfi] {len(jobs)} runs planned, {len(todo)} to run", flush=True)
        for i, (exp, cmd) in enumerate(todo, 1):
            print(f"  [{i}/{len(todo)}] {exp}", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stderr[-1500:])
                raise SystemExit(f"failed: {exp}")

    # ---- aggregate ----
    perf = defaultdict(lambda: defaultdict(list))
    rew = defaultdict(lambda: defaultdict(list))
    cost_share = 0.05
    for atk in args.attacks:
        for meth in args.methods:
            for s in args.seeds:
                run = read_log(args.log_dir, f"dcfi_{meth}_{atk}_seed{s}")
                if not run:
                    continue
                last = run["last"]
                key = (meth, atk)
                for k in ("balanced_acc", "auc", "f1_pos", "client_acc_min", "jain"):
                    if k in last:
                        perf[key][k].append(last[k])
                perf[key]["collapsed"].append(1.0 if last.get("balanced_acc", 1) <= COLLAPSE else 0.0)
                cr = convergence_round(run["evals"])
                if cr is not None:
                    perf[key]["convergence_round"].append(cr)
                if meth == "dcfi":
                    for k, v in reward_metrics(run, cost_share).items():
                        if np.isfinite(v):
                            rew[key][k].append(v)

    L = ["# DCFI-UPA results — dual-channel fair incentive & utility-preserving aggregation\n",
         f"Tabular mental-health protocol, {len(args.seeds)} seeds, adversary "
         f"fraction {args.frac:.0%}. `none` = no adversaries.\n",
         "## Performance channel\n",
         "| Attack | Method | Balanced acc | AUC | F1(pos) | Worst-client | "
         "Conv. round | Collapsed |", "|---|---|---|---|---|---|---|---|"]

    def ms(v, p=3):
        return f"{np.mean(v):.{p}f}±{np.std(v):.{p}f}" if v else "—"

    for atk in args.attacks:
        for meth in args.methods:
            c = perf[(meth, atk)]
            if not c:
                continue
            nc = int(sum(c["collapsed"]))
            L.append(f"| {atk} | {meth} | {ms(c['balanced_acc'])} | {ms(c['auc'])} | "
                     f"{ms(c['f1_pos'])} | {ms(c['client_acc_min'])} | "
                     f"{np.mean(c['convergence_round']):.0f} | {nc}/{len(c['collapsed'])} |")

    L += ["\n## Reward channel (DCFI only)\n",
          "| Attack | Adversary reward share | ρ(reward, LOO) | MAE | "
          "Reward-to-cost | IR rate | Jain(useful rewards) | Jain(selection) | ρ(reward, size) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for atk in args.attacks:
        c = rew[("dcfi", atk)]
        if not c:
            continue
        L.append(f"| {atk} | {np.mean(c['adversary_reward_share'])*100:.2f}% | "
                 f"{ms(c['reward_contrib_spearman'])} | {ms(c['reward_contrib_mae'])} | "
                 f"{ms(c['reward_to_cost'],1)} | {np.mean(c['individual_rationality'])*100:.0f}% | "
                 f"{ms(c['jain_useful_rewards'])} | {ms(c['jain_selection'])} | "
                 f"{ms(c['reward_size_corr'])} |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

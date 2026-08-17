#!/usr/bin/env python3
"""Reward-formula ablations and parameter sensitivity for DCFI-UPA.

The reward rule is a pure function of quantities already logged every round
(per-client utility, fairness contribution, dataset size, active set). So every
ablation of the FORMULA -- dropping a term, changing its weight, changing the
cap or the holdback -- can be recomputed from existing logs without retraining.
Only ablations that change training or selection need new runs.

Ablations answer "does each term earn its place?"; the sweeps answer "how
sensitive is the mechanism to the constants we picked?".

Usage:
    python3 scripts/dcfi_ablation.py
    python3 scripts/dcfi_ablation.py --attacks none zero_update
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STUDY = os.path.join(ROOT, "studies", "tabular_mentalhealth")
sys.path.insert(0, ROOT)


def load(log_dir, exp):
    paths = sorted(glob.glob(os.path.join(log_dir, f"{exp}-*.jsonl")))
    if not paths:
        return None
    rounds, meta = [], {}
    with open(paths[-1]) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "round" and r.get("dcfi"):
                rounds.append(r)
            elif r.get("type") == "client_sizes":
                meta = r
    return dict(rounds=rounds, meta=meta) if rounds else None


def _nz(x, n):
    x = np.clip(np.asarray(x, float), 0.0, None)
    s = x.sum()
    return x / s if s > 1e-12 else np.ones(n) / n


def water_fill_cap(share, cap):
    if cap >= 1.0 or len(share) * cap <= 1.0:
        return share
    s = share.copy()
    for _ in range(len(s)):
        over = s > cap + 1e-12
        if not over.any():
            break
        excess = (s[over] - cap).sum()
        s[over] = cap
        under = ~over
        if not under.any():
            break
        pool = s[under].sum()
        if pool > 1e-12:
            s[under] += excess * (s[under] / pool)
        else:
            # Every uncapped client has zero share (happens for single-term
            # formulas where one client takes the whole round). Spread the
            # excess evenly rather than dividing by zero.
            s[under] += excess / under.sum()
    return s / s.sum()


def simulate(run, w_base, w_util, w_fair, cap, holdback, release_rel=0.7):
    """Recompute cumulative reward under a given formula, from logged signals."""
    n = len(run["meta"]["sizes"])
    reward = np.zeros(n)
    held = np.zeros(n)
    contrib = np.zeros(n)
    selections = np.zeros(n)

    for r in run["rounds"]:
        d = r["dcfi"]
        ids = d.get("kept_ids") or []
        if not ids:
            continue
        u = np.asarray(d.get("utility", []), float)
        f = np.asarray(d.get("fairness", []), float)
        if len(u) != len(ids) or len(f) != len(ids):
            continue
        rel = {int(k): v for k, v in d.get("reliability", {}).items()}
        for cid in r.get("active", []):
            selections[cid] += 1
        contrib[np.array(ids, int)] += u

        m = len(ids)
        share = (w_base * (np.ones(m) / m) + w_util * _nz(u, m) + w_fair * _nz(f, m))
        tot = share.sum()
        share = share / tot if tot > 1e-12 else np.ones(m) / m
        share = water_fill_cap(share, cap)

        for i, cid in enumerate(ids):
            amount = share[i]
            h = amount * holdback
            rel_k = rel.get(int(cid), 1.0)
            release = amount - h
            if rel_k >= release_rel and held[cid] > 0:
                release += held[cid]
                held[cid] = 0.0
            held[cid] += h
            reward[cid] += release
    return reward, contrib, selections


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    def rank(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), float); i = 0; sv = v[o]
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    rx, ry = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def jain(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float((x.sum() ** 2) / (len(x) * (x ** 2).sum())) if len(x) and x.sum() > 0 else np.nan


def metrics(run, reward, contrib, selections, cost_share=0.05):
    meta = run["meta"]
    n = len(meta["sizes"])
    sizes = np.array(meta["sizes"], float)
    adv = set(meta.get("freerider_client_ids", [])) | set(meta.get("noisy_client_ids", []))
    honest = [c for c in range(n) if c not in adv]
    tot = reward.sum()
    cost = selections * cost_share
    with np.errstate(divide="ignore", invalid="ignore"):
        rtc = np.where(cost > 0, reward / cost, np.nan)
    hon = np.array([rtc[c] for c in honest if np.isfinite(rtc[c])])
    return dict(
        adv_share=float(sum(reward[c] for c in adv) / tot) if tot and adv else 0.0,
        fidelity=spearman(reward, contrib),
        rtc=float(np.mean(hon)) if len(hon) else np.nan,
        ir=float(np.mean(hon >= 1.0)) if len(hon) else np.nan,
        jain_reward=jain([reward[c] for c in honest]),
        reward_size_corr=float(np.corrcoef(reward, sizes)[0, 1]) if np.std(reward) > 0 else np.nan,
    )


# (name, w_base, w_util, w_fair) -- weights renormalised so each variant still
# spends exactly the same budget; only the SHAPE of the split changes.
VARIANTS = [
    ("full (0.2/0.5/0.3)",   0.2, 0.5, 0.3),
    ("no base term",         0.0, 0.625, 0.375),
    ("no utility term",      0.4, 0.0, 0.6),
    ("no fairness term",     0.4, 0.6, 0.0),
    ("utility only",         0.0, 1.0, 0.0),
    ("fairness only",        0.0, 0.0, 1.0),
    ("equal split (base only)", 1.0, 0.0, 0.0),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--attacks", nargs="+",
                   default=["none", "zero_update", "copied_update",
                            "scaled_global", "random_update"])
    p.add_argument("--log-dir", default=os.path.join(STUDY, "logs_dcfi"))
    p.add_argument("--out", default=os.path.join(STUDY, "tables", "dcfi_ablation.md"))
    args = p.parse_args()

    runs = defaultdict(list)
    for atk in args.attacks:
        for s in args.seeds:
            r = load(args.log_dir, f"dcfi_dcfi_{atk}_seed{s}")
            if r:
                runs[atk].append(r)
    if not runs:
        raise SystemExit("no DCFI logs found")

    L = ["# DCFI-UPA — reward-formula ablations and parameter sensitivity\n",
         f"Recomputed from {sum(len(v) for v in runs.values())} logged runs "
         "(the reward rule is a pure function of per-round signals already "
         "recorded, so no retraining is needed). Every variant spends the same "
         "budget; only the shape of the split changes.\n"]

    # ---------- A. term ablations ----------
    L.append("\n## A. Does each term in the reward formula earn its place?\n")
    L.append("Averaged over all attack scenarios. `adv share` = fraction of budget "
             "captured by adversaries (lower better; equal-split gives them 40%). "
             "`IR` = share of honest clients paid above cost.\n")
    L.append("| Reward formula | Adv share | Fidelity ρ | Reward/cost | IR | Jain(reward) | ρ(reward,size) |")
    L.append("|---|---|---|---|---|---|---|")
    for name, wb, wu, wf in VARIANTS:
        acc = defaultdict(list)
        for atk, rs in runs.items():
            for run in rs:
                rw, ct, sel = simulate(run, wb, wu, wf, cap=0.4, holdback=0.2)
                for k, v in metrics(run, rw, ct, sel).items():
                    if np.isfinite(v):
                        acc[k].append(v)
        L.append(f"| {name} | {np.mean(acc['adv_share'])*100:.2f}% | "
                 f"{np.mean(acc['fidelity']):+.3f} | {np.mean(acc['rtc']):.1f} | "
                 f"{np.mean(acc['ir'])*100:.0f}% | {np.mean(acc['jain_reward']):.3f} | "
                 f"{np.mean(acc['reward_size_corr']):+.3f} |")

    # ---------- B. parameter sweeps ----------
    def sweep(title, note, param, values, build):
        L.append(f"\n## {title}\n")
        L.append(note + "\n")
        L.append(f"| {param} | Adv share | Fidelity ρ | Reward/cost | IR | Jain(reward) |")
        L.append("|---|---|---|---|---|---|")
        for v in values:
            kw = build(v)
            acc = defaultdict(list)
            for atk, rs in runs.items():
                for run in rs:
                    rw, ct, sel = simulate(run, **kw)
                    for k, val in metrics(run, rw, ct, sel).items():
                        if np.isfinite(val):
                            acc[k].append(val)
            L.append(f"| {v} | {np.mean(acc['adv_share'])*100:.2f}% | "
                     f"{np.mean(acc['fidelity']):+.3f} | {np.mean(acc['rtc']):.1f} | "
                     f"{np.mean(acc['ir'])*100:.0f}% | {np.mean(acc['jain_reward']):.3f} |")

    sweep("B. Base-payment weight (individual rationality vs efficiency)",
          "The base term pays every participant regardless of measured utility. "
          "Raising it protects individual rationality; lowering it concentrates "
          "reward on high-utility clients but also on anyone who fakes utility.",
          "w_base", [0.0, 0.1, 0.2, 0.3, 0.5],
          lambda v: dict(w_base=v, w_util=(1 - v) * 0.625, w_fair=(1 - v) * 0.375,
                         cap=0.4, holdback=0.2))

    sweep("C. Reward cap (concentration control)",
          "Water-filling cap on any single client's share of a round's budget. "
          "1.0 disables it.",
          "cap", [0.2, 0.3, 0.4, 0.6, 1.0],
          lambda v: dict(w_base=0.2, w_util=0.5, w_fair=0.3, cap=v, holdback=0.2))

    sweep("D. Holdback fraction (deferred settlement)",
          "Fraction withheld each round and released only once the client's "
          "reliability clears the threshold.",
          "holdback", [0.0, 0.1, 0.2, 0.3, 0.5],
          lambda v: dict(w_base=0.2, w_util=0.5, w_fair=0.3, cap=0.4, holdback=v))

    sweep("E. Utility/fairness balance (efficiency vs redistribution)",
          "Splitting the non-base budget between measured utility and "
          "minority-class fairness contribution.",
          "w_utility (rest to fairness)", [0.0, 0.2, 0.4, 0.6, 0.8],
          lambda v: dict(w_base=0.2, w_util=0.8 * v, w_fair=0.8 * (1 - v),
                         cap=0.4, holdback=0.2))

    # ---------- E2. participation-cost sweep ----------
    # IR reads 100% everywhere at the default cost, which means the cost is not
    # binding rather than that the mechanism is generous. Sweeping it finds the
    # break-even point and turns IR into an informative metric.
    L.append("\n## E2. Participation cost (where does individual rationality break?)\n")
    L.append("Cost is expressed as a share of the per-round budget, charged per "
             "round a client is selected. IR = fraction of honest clients whose "
             "cumulative reward covers their cumulative cost.\n")
    L.append("| cost/round | Reward-to-cost | IR (honest) | IR (w_base=0) |")
    L.append("|---|---|---|---|")
    for cost in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc, acc0 = defaultdict(list), defaultdict(list)
        for atk, rs in runs.items():
            for run in rs:
                rw, ct, sel = simulate(run, 0.2, 0.5, 0.3, cap=0.4, holdback=0.2)
                for k, v in metrics(run, rw, ct, sel, cost_share=cost).items():
                    if np.isfinite(v):
                        acc[k].append(v)
                rw0, ct0, sel0 = simulate(run, 0.0, 0.625, 0.375, cap=0.4, holdback=0.2)
                for k, v in metrics(run, rw0, ct0, sel0, cost_share=cost).items():
                    if np.isfinite(v):
                        acc0[k].append(v)
        L.append(f"| {cost:.2f} | {np.mean(acc['rtc']):.2f} | "
                 f"{np.mean(acc['ir'])*100:.0f}% | {np.mean(acc0['ir'])*100:.0f}% |")

    # ---------- F. per-scenario breakdown of the chosen formula ----------
    L.append("\n## F. Chosen formula, per scenario\n")
    L.append("| Scenario | Adv share | Fidelity ρ | Reward/cost | IR | Jain(reward) | ρ(reward,size) |")
    L.append("|---|---|---|---|---|---|---|")
    for atk in args.attacks:
        if atk not in runs:
            continue
        acc = defaultdict(list)
        for run in runs[atk]:
            rw, ct, sel = simulate(run, 0.2, 0.5, 0.3, cap=0.4, holdback=0.2)
            for k, v in metrics(run, rw, ct, sel).items():
                if np.isfinite(v):
                    acc[k].append(v)
        L.append(f"| {atk} | {np.mean(acc['adv_share'])*100:.2f}% | "
                 f"{np.mean(acc['fidelity']):+.3f} | {np.mean(acc['rtc']):.1f} | "
                 f"{np.mean(acc['ir'])*100:.0f}% | {np.mean(acc['jain_reward']):.3f} | "
                 f"{np.mean(acc['reward_size_corr']):+.3f} |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

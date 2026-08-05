#!/usr/bin/env python3
"""Aggregate the tabular mental-health sweep into tables + the mechanism analysis.

Produces three things, all read at the LAST (converged) round -- no best-round
cherry-picking:

  1. Per-method results by noise level (balanced accuracy is the headline, since
     the depression target is 58.6% positive).
  2. The "opening" analysis: how much noise costs the strongest baseline, versus
     how much the proposed method recovers. This is the §5.1 diagnostic re-run.
  3. The MECHANISM / RANKING test: does the loss detector give corrupted clients
     less aggregation weight than clean ones? This is the pre-registered primary
     question (docs/INCENTIVE_PROPOSAL.md §10.2) -- it is what must transfer from
     the imaging study for the incentive direction to be viable.

Usage:  python3 scripts/aggregate_tabular.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NAME_RE = re.compile(r"^(?P<ds>.+?)_(?P<regime>screening|full)_(?P<method>[a-z_]+)"
                     r"_n(?P<noise>\d+)_seed(?P<seed>\d+)$")


def load_runs(log_dir: str):
    runs = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.jsonl"))):
        exp = os.path.basename(path).rsplit("-", 2)[0]
        m = NAME_RE.match(exp)
        if not m:
            continue
        evals, rounds, noisy_ids, cfg = [], [], [], {}
        with open(path) as f:
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
                    noisy_ids = rec.get("noisy_client_ids", [])
                elif t == "config":
                    cfg = rec
        if not evals:
            continue
        runs.append(dict(
            exp=exp, method=m["method"], regime=m["regime"],
            noise=int(m["noise"]) / 100.0, seed=int(m["seed"]),
            last=evals[-1], rounds=rounds, noisy_ids=noisy_ids, cfg=cfg,
        ))
    return runs


def _ms(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), a.std(), len(a)


def results_table(runs, metric_keys=("balanced_acc", "acc", "auc", "jain",
                                     "client_acc_min", "ece", "brier", "auprc",
                                     "f1_pos", "bacc_opt")):
    """method x noise -> mean±std over seeds, at the last round."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for k in metric_keys:
            v = r["last"].get(k)
            if v is not None:
                cells[(r["method"], r["noise"])][k].append(v)
    return cells


def mechanism(runs):
    """Mean aggregation weight given to corrupted vs clean clients.

    Weights are per-round over the ACTIVE set only, so we pair each round's
    `active` list with its `weights` list and bucket by ground-truth corruption.
    A ratio < 1 means the detector correctly down-weights corrupted clients.
    """
    out = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["noise"] == 0.0 or not r["noisy_ids"]:
            continue
        noisy = set(r["noisy_ids"])
        w_noisy, w_clean = [], []
        for rd in r["rounds"]:
            active, weights = rd.get("active"), rd.get("weights")
            if not active or not weights or len(active) != len(weights):
                continue
            for cid, w in zip(active, weights):
                (w_noisy if cid in noisy else w_clean).append(float(w))
        if w_noisy and w_clean:
            out[(r["method"], r["noise"])]["noisy"].append(float(np.mean(w_noisy)))
            out[(r["method"], r["noise"])]["clean"].append(float(np.mean(w_clean)))
    return out


def detection_auc(runs):
    """Can the per-round probe loss separate corrupted from clean clients?

    Pools every (client, round) probe-loss observation in a run and asks how well
    a higher loss predicts "this client is corrupted" (rank AUC). This is the
    ranking property the incentive mechanism actually depends on, measured
    directly rather than via its effect on accuracy.
    """
    from federated.server import _binary_auc
    out = defaultdict(list)
    for r in runs:
        if r["noise"] == 0.0 or not r["noisy_ids"]:
            continue
        noisy = set(r["noisy_ids"])
        losses, labels = [], []
        for rd in r["rounds"]:
            active, cl = rd.get("active"), rd.get("contrib_losses")
            if not active or not cl or len(active) != len(cl):
                continue
            for cid, loss in zip(active, cl):
                losses.append(float(loss))
                labels.append(1 if cid in noisy else 0)
        if losses and 0 < sum(labels) < len(labels):
            out[(r["method"], r["noise"])].append(
                _binary_auc(np.array(labels), np.array(losses))
            )
    return out


def tost_noninferiority(prop, base, margin=0.01):
    """One-sided non-inferiority test: is `prop` >= `base` - margin?

    Welch t-test on H0: mean(prop) - mean(base) <= -margin. Returns (delta, p).
    Small p => non-inferiority established at the given margin.
    """
    from scipy import stats
    a, b = np.array(prop, float), np.array(base, float)
    delta = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return delta, 0.0 if delta > -margin else 1.0
    df = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
        + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    t = (delta + margin) / se
    return delta, float(1 - stats.t.cdf(t, df))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir",
                   default=os.path.join(ROOT, "studies", "tabular_mentalhealth", "logs"))
    p.add_argument("--out", default=os.path.join(
        ROOT, "studies", "tabular_mentalhealth", "tables", "tabular_tables.md"))
    args = p.parse_args()

    runs = load_runs(args.log_dir)
    print(f"loaded {len(runs)} runs from {args.log_dir}")
    if not runs:
        return

    methods = ["fedavg", "fedprox", "feddyn", "fedce", "proposedcssv", "proposed"]
    noises = sorted({r["noise"] for r in runs})
    cells = results_table(runs)
    mech = mechanism(runs)
    dauc = detection_auc(runs)

    L = []
    ds = runs[0]["exp"].split("_screening")[0]
    n_seeds = len({r["seed"] for r in runs})
    L.append(f"# Tabular mental-health results — {ds} (screening regime)\n")
    L.append(f"Last-round metrics, mean±std over {n_seeds} seeds. Frozen protocol: 10 clients, "
             "5 active/round, 2 local epochs, 40 rounds, SGD lr=0.005 cosine-decayed, "
             "Dirichlet α=0.3. **Balanced accuracy is the headline** (target is 58.6% "
             "positive, so plain accuracy flatters majority-class prediction).\n")
    L.append("Reference points: majority-class = 0.586 acc / 0.500 balanced acc; "
             "centralized pooled-data ceiling = 0.794 acc / 0.783 balanced acc.\n")

    for n in noises:
        label = "Clean (0% noisy clients)" if n == 0 else \
                f"{int(n*100)}% noisy clients (60% label flip)"
        L.append(f"\n### {label}\n")
        L.append("| Method | Balanced acc | Accuracy | AUC | Jain | Worst-client | Collapsed seeds |")
        L.append("|---|---|---|---|---|---|---|")
        for m in methods:
            c = cells.get((m, n))
            if not c:
                continue
            def f(k, prec=3):
                if k not in c or not c[k]:
                    return "—"
                mu, sd, cnt = _ms(c[k])
                return f"{mu:.{prec}f}±{sd:.{prec}f}"
            # A "collapsed" run is one whose balanced accuracy landed at chance
            # (<=0.51), i.e. the model predicts a single class. These are the
            # dominant source of variance at high noise and are worth surfacing
            # rather than hiding inside a large std.
            bacc = c.get("balanced_acc", [])
            ncol = sum(1 for v in bacc if v <= 0.51)
            L.append(f"| {m} | {f('balanced_acc')} | {f('acc')} | {f('auc')} | "
                     f"{f('jain')} | {f('client_acc_min')} | {ncol}/{len(bacc)} |")

    # --- beyond-accuracy metrics ---
    # Accuracy and balanced accuracy both read the model at the DEFAULT 0.5
    # threshold, so they conflate "cannot separate the classes" with "separates
    # them but the probabilities are mis-scaled". These columns pull those apart.
    L.append("\n## Beyond accuracy: calibration, ranking quality, and threshold effects\n")
    L.append("`ece`/`brier` = calibration (lower better) — how trustworthy the predicted "
             "probability is, which is what matters for clinical triage. `auprc` = ranking "
             "quality on the positive (depressed) class. `bacc_opt` = balanced accuracy at "
             "the BEST threshold; the gap `bacc_opt − balanced_acc` is the loss due purely "
             "to a mis-placed operating point rather than to weaker discrimination.\n")
    for n in noises:
        label = "Clean" if n == 0 else f"{int(n*100)}% noisy clients"
        L.append(f"\n### {label}\n")
        L.append("| Method | ECE ↓ | Brier ↓ | AUPRC ↑ | F1(pos) ↑ | bacc_opt ↑ | "
                 "bacc_opt − balanced_acc |")
        L.append("|---|---|---|---|---|---|---|")
        for m in methods:
            c = cells.get((m, n))
            if not c or "ece" not in c or not c["ece"]:
                continue

            def g(k, prec=3):
                if k not in c or not c[k]:
                    return "—"
                mu, sd, _ = _ms(c[k])
                return f"{mu:.{prec}f}±{sd:.{prec}f}"
            gap = ""
            if c.get("bacc_opt") and c.get("balanced_acc"):
                gap = f"{_ms(c['bacc_opt'])[0] - _ms(c['balanced_acc'])[0]:+.3f}"
            L.append(f"| {m} | {g('ece')} | {g('brier')} | {g('auprc')} | "
                     f"{g('f1_pos')} | {g('bacc_opt')} | {gap} |")

    # --- the "opening" analysis (§5.1 diagnostic, re-run on tabular) ---
    L.append("\n## The \"opening\": how much does noise cost each baseline?\n")
    L.append("| Method | Clean bal-acc | 40% bal-acc | Degradation |")
    L.append("|---|---|---|---|")
    for m in methods:
        c0, c4 = cells.get((m, 0.0)), cells.get((m, 0.4))
        if not c0 or not c4:
            continue
        a, b = _ms(c0["balanced_acc"])[0], _ms(c4["balanced_acc"])[0]
        L.append(f"| {m} | {a:.3f} | {b:.3f} | {a-b:+.3f} |")

    # --- non-inferiority of proposed vs the strongest baseline ---
    L.append("\n## Non-inferiority: proposed vs FedDyn (margin 0.01 balanced acc)\n")
    L.append("| Noise | Δ (proposed − FedDyn) | TOST p | Non-inferior at 0.01? |")
    L.append("|---|---|---|---|")
    for n in noises:
        cp, cf = cells.get(("proposed", n)), cells.get(("feddyn", n))
        if not cp or not cf:
            continue
        try:
            d, pv = tost_noninferiority(cp["balanced_acc"], cf["balanced_acc"])
            L.append(f"| {int(n*100)}% | {d:+.4f} | {pv:.4f} | "
                     f"{'**yes**' if pv < 0.05 else 'not established'} |")
        except Exception as e:  # scipy missing
            L.append(f"| {int(n*100)}% | (test unavailable: {e}) | | |")

    # --- seed-matched head-to-head (robust to the collapse-driven variance) ---
    L.append("\n## Seed-matched head-to-head: proposed vs FedDyn\n")
    L.append("Same seed = same data partition and same corrupted-client set, so this "
             "pairs runs rather than comparing noisy means.\n")
    L.append("| Noise | Proposed wins | Mean Δ | Median Δ |")
    L.append("|---|---|---|---|")
    for n in noises:
        fd = {r["seed"]: r["last"]["balanced_acc"]
              for r in runs if r["method"] == "feddyn" and r["noise"] == n}
        pr = {r["seed"]: r["last"]["balanced_acc"]
              for r in runs if r["method"] == "proposed" and r["noise"] == n}
        common = sorted(set(fd) & set(pr))
        if not common:
            continue
        d = np.array([pr[s] - fd[s] for s in common])
        L.append(f"| {int(n*100)}% | {int((d > 0).sum())}/{len(common)} | "
                 f"{d.mean():+.4f} | {np.median(d):+.4f} |")

    # --- THE key test: does the ranking property transfer? ---
    L.append("\n## Mechanism / ranking test (the pre-registered primary question)\n")
    L.append("Mean aggregation weight given to corrupted vs clean clients, pooled over "
             "all rounds and seeds. **Ratio < 1 means corrupted clients are correctly "
             "down-weighted.** FedAvg/FedDyn have no quality signal (size- or "
             "uniform-weighted), so they are the control.\n")
    L.append("| Method | Noise | Weight: corrupted | Weight: clean | Ratio | Probe-loss detection AUC |")
    L.append("|---|---|---|---|---|---|")
    for m in methods:
        for n in noises:
            if n == 0:
                continue
            d = mech.get((m, n))
            if not d:
                continue
            wn, wc = np.mean(d["noisy"]), np.mean(d["clean"])
            au = dauc.get((m, n))
            aus = f"{np.mean(au):.3f}±{np.std(au):.3f}" if au else "—"
            L.append(f"| {m} | {int(n*100)}% | {wn:.4f} | {wc:.4f} | "
                     f"**{wn/wc:.3f}** | {aus} |")

    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt)
    print(txt)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    main()

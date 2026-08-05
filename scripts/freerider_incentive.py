#!/usr/bin/env python3
"""Criterion C2/C3: do free-riders get starved of reward, and which signal sees them?

Pre-registered bars (docs/INCENTIVE_PROPOSAL.md §10.1):
    C2 - free-riders capture <= 5% of cumulative budget at a 40% free-rider
         fraction (equal-split captures ~40%), for rungs f1-f4.
    C3 - free-rider detection AUC >= 0.85 on rungs f1-f4.

Why this is a different experiment from C1
------------------------------------------
The loss detector probes the *global model's loss on a client's own data*. A
free-rider that holds perfectly good data but does not train still has a LOW
probe loss, so the loss signal cannot see it -- it measures data quality, not
effort. Effort shows up in the UPDATE instead. So this script scores every
client with three complementary signals, computed from the same logged runs:

    loss  q ∝ exp(-beta * (loss_k - min_loss))     <- catches label noise
    cssv  normalized ShapFed cosine score          <- catches wrong DIRECTION
    norm  penalises |log(||update|| / median)|     <- catches wrong MAGNITUDE

and a blend of all three (the §8.1 score, generalised). The attack ladder is
built so the signals dissociate:

    f1 random update      -> wrong direction, small norm
    f2 zero effort        -> zero norm, no direction at all
    f3 trains on 10%      -> right direction, small norm    (cosine is blind)
    f4 inflates update x3 -> right direction, large norm    (cosine is BLIND by
                             construction: cosine is scale-invariant)

Usage:
    python3 scripts/freerider_incentive.py
    python3 scripts/freerider_incentive.py --seeds 0 1 --modes f2 f4
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
sys.path.insert(0, ROOT)

PROTOCOL = dict(
    dataset="student_depression", feature_regime="screening",
    num_clients=10, candidate_size_d=10, active_size_m=5, num_rounds=40,
    local_epochs=2, local_lr=0.005, batch_size=128, alpha=0.3,
    local_test_frac=0.2, lr_schedule="cosine", num_workers=0,
    hidden_dim=64, mlp_depth=2, mlp_dropout=0.1,
)
LOSS_BETA = 1.5
NORM_GAMMA = 2.0     # sharpness of the norm-deviation penalty
# Effort-gate tolerance, in log space. log(2)=0.69 admits any client within a
# factor of 2 of the peer median effort density in either direction -- wide
# enough not to punish ordinary non-IID variation (measured honest spread:
# CV 0.69), tight enough to exclude a 3x inflation. Chosen from the honest
# spread, not tuned against the attack.
GATE_LOG_TOL = 0.693
EPS = 1e-9


def build_cmd(seed, mode, frac, exp, log_dir):
    cmd = [sys.executable, os.path.join(ROOT, "main.py"),
           "--method", "proposed", "--seed", str(seed), "--exp-name", exp,
           "--device", "cpu", "--log-dir", log_dir, "--eval-every", "40",
           "--freerider-mode", mode, "--freerider-frac", str(frac)]
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
    return dict(last=evals[-1] if evals else {}, rounds=rounds, meta=meta)


def _norm(x):
    x = np.asarray(x, float)
    s = x.sum()
    return x / s if s > 1e-12 else np.ones_like(x) / max(len(x), 1)


def round_scores(rd, sizes, history=None):
    """Per-active-client score under each signal, for one round.

    ``history`` is a mutable list of effort densities seen in earlier rounds. A
    real server accumulates this, and it matters: the reference for "normal
    effort" taken from only the 5 active clients in one round is far too noisy
    to see a 3x inflation, whereas a running median over history is stable.
    """
    active = rd.get("active") or []
    if not active:
        return None
    a = np.array(active, int)
    out = {"equal": np.ones(len(a)) / len(a), "size": _norm(sizes[a])}

    losses = rd.get("contrib_losses")
    if losses and len(losses) == len(a):
        lo = np.asarray(losses, float)
        out["loss"] = _norm(np.exp(-LOSS_BETA * (lo - lo.min())))

    cs = rd.get("diag_cssv_score")
    if cs and len(cs) == len(a):
        out["cssv"] = _norm(np.clip(np.asarray(cs, float), 0, None))

    nm = rd.get("update_norms")
    if nm and len(nm) == len(a):
        v = np.asarray(nm, float)

        def two_sided(x, ref=None):
            """Penalise deviation from a reference in EITHER direction -- too
            small (f1/f2/f3, not working) and too large (f4, inflating)."""
            if ref is None:
                pos = x[x > EPS]
                ref = np.median(pos) if len(pos) else 1.0
            return _norm(np.exp(-NORM_GAMMA * np.abs(np.log((x + EPS) / (ref + EPS)))))

        # Raw norm. Kept to DEMONSTRATE its flaw: honest ||update|| scales ~linearly
        # with client size (measured: size^0.87, corr +0.40), so a small client's
        # inflated update looks like a large client's honest one.
        out["norm"] = two_sided(v)
        # Effort density = movement per sample. Removes the size confound, since
        # local steps are proportional to n_k at fixed epochs.
        dens = v / np.maximum(sizes[a], 1.0)
        # Reference = median over THIS round's active clients. A pooled
        # running median was tried and is worse: with a 40% free-rider
        # population every round contributes inflated values, so the history
        # itself becomes ~40% contaminated and the reference drifts toward the
        # attack. The per-round median over 5 clients is noisier but far less
        # biased, because a minority of the active set is usually adversarial.
        if history is not None:
            history.extend(dens.tolist())
        out["effort"] = two_sided(dens)

    if all(k in out for k in ("loss", "cssv", "effort")):
        out["blend"] = _norm(out["loss"] + out["cssv"] + out["effort"])
        # The recommended rule: quality x effort. Multiplicative, so a client must
        # be BOTH honest-looking (low loss) AND actually working (normal effort
        # density) to be paid. Either failure alone drives the reward toward zero.
        out["blend_mult"] = _norm(out["loss"] * out["effort"])

        # Reputation-GATED variant (§8.2): a hard eligibility test before any
        # money is split. A soft blend can be out-voted -- an f4 client looks
        # excellent on the loss term (its data is genuine) and that can outweigh
        # the effort penalty, so inflation actually PAYS. The gate removes that
        # trade: fail the effort test and the reward is zero regardless of how
        # good the data looks. Budget is then split among survivors only, so the
        # rule stays strongly budget-balanced.
        pos = dens[dens > EPS]
        ref = float(np.median(pos)) if len(pos) else 1.0
        dev = np.abs(np.log((dens + EPS) / (ref + EPS)))
        eligible = (dev <= GATE_LOG_TOL).astype(float)
        out["gated"] = (_norm(out["loss"] * eligible) if eligible.sum() > 0
                        else np.zeros(len(a)))
    return a, out


RULES = ["gated", "blend_mult", "blend", "loss", "cssv", "effort", "norm", "size", "equal"]


def analyse(run, n_clients):
    """Cumulative reward share per rule, and per-signal detection AUC."""
    sizes = np.array(run["meta"]["sizes"], float)
    fr = set(run["meta"].get("freerider_client_ids", []))
    reward = {r: np.zeros(n_clients) for r in RULES}
    seen = np.zeros(n_clients)                  # rounds each client was active
    obs = {r: ([], []) for r in RULES}          # (per-round scores, is_freerider)
    history: list = []

    for rd in run["rounds"]:
        got = round_scores(rd, sizes, history)
        if got is None:
            continue
        a, sc = got
        seen[a] += 1
        for r in RULES:
            if r in sc:
                reward[r][a] += sc[r]
                obs[r][0].extend(sc[r].tolist())
                obs[r][1].extend([(1 if c in fr else 0) for c in a])

    from federated.server import _binary_auc
    lab_c = np.array([1 if c in fr else 0 for c in range(n_clients)])
    out = {}
    for r in RULES:
        tot = reward[r].sum()
        share = float(sum(reward[r][c] for c in fr) / tot) if tot > 0 and fr else np.nan
        s, lab = np.array(obs[r][0]), np.array(obs[r][1])
        # A good signal gives free-riders LOW scores, so (-signal) predicts
        # "is a free-rider"; AUC 1.0 = perfect separation.
        auc = (_binary_auc(lab, -s) if len(s) and 0 < lab.sum() < len(lab) else np.nan)
        # Decision-relevant version: the mechanism pays on CUMULATIVE score, not
        # per round, so it gets to accumulate evidence across a client's rounds.
        # Normalised by participation so it measures rate, not luck of selection.
        rate = reward[r] / np.maximum(seen, 1)
        auc_c = (_binary_auc(lab_c, -rate)
                 if 0 < lab_c.sum() < n_clients else np.nan)
        out[r] = dict(freerider_share=share, detection_auc=float(auc),
                      detection_auc_cumulative=float(auc_c))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--modes", nargs="+", default=["f1", "f2", "f3", "f4"])
    p.add_argument("--fracs", nargs="+", type=float, default=[0.4])
    p.add_argument("--log-dir", default=os.path.join(STUDY, "logs_freerider"))
    p.add_argument("--out", default=os.path.join(STUDY, "tables", "freerider_incentive.md"))
    p.add_argument("--json-out", default=os.path.join(STUDY, "tables", "freerider_incentive.json"))
    p.add_argument("--skip-runs", action="store_true")
    args = p.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    N = PROTOCOL["num_clients"]

    jobs = []
    for frac in args.fracs:
        for mode in args.modes:
            for seed in args.seeds:
                exp = f"fr_{mode}_p{round(frac*100):02d}_seed{seed}"
                jobs.append((exp, build_cmd(seed, mode, frac, exp, args.log_dir)))

    if not args.skip_runs:
        todo = [(e, c) for e, c in jobs
                if not glob.glob(os.path.join(args.log_dir, f"{e}-*.jsonl"))]
        print(f"[c2] {len(jobs)} runs planned, {len(todo)} to run")
        for i, (exp, cmd) in enumerate(todo, 1):
            print(f"  [{i}/{len(todo)}] {exp}", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-1500:]); print(r.stderr[-1500:])
                raise SystemExit(f"failed: {exp}")

    results = {}
    for frac in args.fracs:
        for mode in args.modes:
            per_seed = []
            for seed in args.seeds:
                run = read_log(args.log_dir, f"fr_{mode}_p{round(frac*100):02d}_seed{seed}")
                if run and run["rounds"]:
                    per_seed.append(analyse(run, N))
            if not per_seed:
                continue
            agg = {}
            for r in RULES:
                def col(key):
                    return [d[r][key] for d in per_seed if np.isfinite(d[r][key])]
                sh, au, auc_c = (col("freerider_share"), col("detection_auc"),
                                 col("detection_auc_cumulative"))
                agg[r] = dict(share=float(np.mean(sh)) if sh else np.nan,
                              share_std=float(np.std(sh)) if sh else np.nan,
                              auc=float(np.mean(au)) if au else np.nan,
                              auc_std=float(np.std(au)) if au else np.nan,
                              auc_cum=float(np.mean(auc_c)) if auc_c else np.nan,
                              auc_cum_std=float(np.std(auc_c)) if auc_c else np.nan)
            results[f"{mode}_p{round(frac*100):02d}"] = agg

    L = ["# C2/C3 — free-rider reward starvation and detection\n",
         f"{len(args.seeds)} seeds, clean labels (no label noise), free-rider "
         f"fraction {args.fracs}. Budget B=1 per round, split among that round's "
         "active clients by each rule, accumulated over 40 rounds.\n",
         "**Pre-registered bars:** C2 — free-rider budget share ≤ 5% "
         "(equal-split ≈ 40%). C3 — detection AUC ≥ 0.85.\n"]

    for key, agg in results.items():
        mode, frac = key.split("_p")
        L.append(f"\n### Rung {mode.upper()} — {int(frac)}% free-riders\n")
        L.append("| Reward signal | Free-rider budget share | Detection AUC "
                 "(per round) | Detection AUC (cumulative) | C2 | C3 |")
        L.append("|---|---|---|---|---|---|")
        for r in RULES:
            d = agg[r]
            if not np.isfinite(d["share"]):
                continue
            c2 = "**PASS**" if d["share"] <= 0.05 else "fail"
            c3 = "**PASS**" if d["auc_cum"] >= 0.85 else "fail"
            L.append(f"| {r} | {d['share']*100:.1f}% ± {d['share_std']*100:.1f} | "
                     f"{d['auc']:.3f} ± {d['auc_std']:.3f} | "
                     f"{d['auc_cum']:.3f} ± {d['auc_cum_std']:.3f} | {c2} | {c3} |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    json.dump(results, open(args.json_out, "w"), indent=2)
    print("\n".join(L))
    print(f"\nwrote {args.out}\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

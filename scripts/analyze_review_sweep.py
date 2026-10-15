#!/usr/bin/env python3
"""Mechanism-level analysis of the review sweep.

Two questions the reviews ask that an accuracy table cannot answer:

  A.3  "provide mechanism-level results for all three datasets" -- does the
       detector actually down-weight corrupted clients, and by how much?

  A.2  "clarify how the method distinguishes corrupted data from novel,
       difficult, or distributionally shifted but correctly labeled data" --
       when a client is merely DIFFERENT, does it get punished like a client
       that is WRONG? If the two are indistinguishable to the score, the
       method removes useful information, which is the failure the reviewer
       suspects.

The sweep stores, for every round, the aggregation weights and the active
client ids, alongside ground truth for which clients were corrupted and which
were covariate-shifted. That is enough to compute both.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "studies" / "review"


def load(dataset: str, tiers) -> list[dict]:
    rows = []
    for t in tiers:
        p = OUT / f"{dataset}_tier{t}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not r.get("failed"):
                    rows.append(r)
    return rows


def weight_ratio(rec: dict, group_key: str):
    """Mean weight of `group` clients divided by mean weight of plain clients.

    Weights are normalised per round, so they are only comparable within a
    round; we therefore compute the ratio round by round and average the
    ratios, never the raw weights. Rounds where the group is absent from the
    active set contribute nothing.
    """
    group = set(rec.get(group_key) or [])
    other_bad = set()
    for k in ("noisy_client_ids", "shifted_client_ids", "freerider_client_ids"):
        if k != group_key:
            other_bad |= set(rec.get(k) or [])
    if not group:
        return None, 0
    ratios = []
    for rd in rec.get("weights_by_round") or []:
        act, w = rd.get("active"), rd.get("weights")
        if not act or not w or len(act) != len(w):
            continue
        w = np.asarray(w, dtype=float)
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        g = np.array([c in group for c in act])
        # "clean" means neither this group nor any other injected condition
        c = np.array([(c not in group) and (c not in other_bad) for c in act])
        if g.sum() == 0 or c.sum() == 0:
            continue
        ratios.append(w[g].mean() / max(w[c].mean(), 1e-12))
    if not ratios:
        return None, 0
    return float(np.mean(ratios)), len(ratios)


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["aptos", "isic", "octmnist"])
    ap.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()

    for ds in args.datasets:
        rows = load(ds, args.tiers)
        if not rows:
            continue
        print(f"\n{'='*78}\n{ds.upper()}   ({len(rows)} completed runs)\n{'='*78}")

        # ---- accuracy by corruption level x method (A.3, B.2) -------------
        noise = [r for r in rows if r["axis"] == "noise"]
        if noise:
            lv = sorted({r["overrides"]["noisy_client_frac"] for r in noise})
            ms = sorted({r["method"] for r in noise},
                        key=lambda m: (m != "proposed", m))
            print("\nACCURACY by corruption level  (rows = method, cols = % clients corrupted)")
            print(f"{'method':<12}" + "".join(f"{int(l*100):>9}%" for l in lv))
            for m in ms:
                line = f"{m:<12}"
                for l in lv:
                    v = [r.get("acc") for r in noise
                         if r["method"] == m and r["overrides"]["noisy_client_frac"] == l
                         and r.get("acc") is not None]
                    line += f"{(np.mean(v) if v else float('nan')):>10.3f}"
                print(line)
            # delta vs FedDyn, which is the comparison the paper claims
            print(f"\n{'delta vs FedDyn':<12}" + "".join(f"{int(l*100):>9}%" for l in lv))
            for m in ms:
                if m == "feddyn":
                    continue
                line = f"{m:<12}"
                for l in lv:
                    a = [r.get("acc") for r in noise if r["method"] == m
                         and r["overrides"]["noisy_client_frac"] == l and r.get("acc") is not None]
                    b = [r.get("acc") for r in noise if r["method"] == "feddyn"
                         and r["overrides"]["noisy_client_frac"] == l and r.get("acc") is not None]
                    line += (f"{np.mean(a)-np.mean(b):>+10.3f}" if a and b else f"{'—':>10}")
                print(line)

        # ---- mechanism: who gets down-weighted? (A.3) ----------------------
        print("\nMECHANISM — mean aggregation weight relative to clean clients")
        print("  (<1 means the group is down-weighted; 1.0 means treated identically)")
        print(f"{'axis':<8}{'method':<12}{'corrupted':>11}{'shifted':>10}"
              f"{'poisoner':>10}  rounds")
        seen = set()
        for r in rows:
            key = (r["axis"], r["method"], json.dumps(r["overrides"], sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            c, nc = weight_ratio(r, "noisy_client_ids")
            s, ns = weight_ratio(r, "shifted_client_ids")
            p, np_ = weight_ratio(r, "freerider_client_ids")
            if c is None and s is None and p is None:
                continue
            print(f"{r['axis']:<8}{r['method']:<12}{fmt(c):>11}{fmt(s):>10}"
                  f"{fmt(p):>10}  {max(nc, ns, np_)}")

        # ---- A.2: corrupted vs shifted, side by side -----------------------
        mixed = [r for r in rows if r["axis"] == "mixed"]
        if mixed:
            print("\nREVIEWER A.2 — corrupted AND shifted clients present in the same run")
            print("  A method that tells them apart down-weights 'corrupted' well below")
            print("  'shifted'. Equal treatment means the score cannot distinguish them.")
            print(f"{'method':<12}{'corrupted':>11}{'shifted':>10}{'separation':>12}")
            for r in mixed:
                c, _ = weight_ratio(r, "noisy_client_ids")
                s, _ = weight_ratio(r, "shifted_client_ids")
                sep = (s - c) if (c is not None and s is not None) else None
                print(f"{r['method']:<12}{fmt(c):>11}{fmt(s):>10}{fmt(sep):>12}")

        # ---- other axes ----------------------------------------------------
        for axis, label in (("type", "NOISE TYPE (asymmetric pair-flip)"),
                            ("shift", "COVARIATE SHIFT ONLY (labels correct)"),
                            ("poison", "MODEL POISONING"),
                            ("alpha", "NON-IID DEGREE"),
                            ("part", "PARTICIPATION RATIO"),
                            ("scale", "CLIENT COUNT")):
            sub = [r for r in rows if r["axis"] == axis]
            if not sub:
                continue
            print(f"\n{label}")
            for r in sorted(sub, key=lambda x: (x["method"], str(x["overrides"]))):
                ov = {k: v for k, v in r["overrides"].items()
                      if k in ("alpha", "active_size_m", "num_clients",
                               "freerider_mode", "label_noise_type",
                               "shift_severity", "noisy_client_frac")}
                print(f"  {r['method']:<12}{str(ov):<70} acc={fmt(r.get('acc'))}")


if __name__ == "__main__":
    main()

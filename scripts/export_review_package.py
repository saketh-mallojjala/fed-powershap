#!/usr/bin/env python3
"""Export the revision evidence as one shareable folder, organised by review point.

Every file is named for the reviewer comment it answers, so the professor (and
later a reviewer) can go from a comment straight to the number that addresses
it without reading the code.
"""
from __future__ import annotations

import csv, json, math, shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "studies" / "review"
OUT = ROOT / "review_package"
METHODS = ["proposed", "feddyn", "fednoro", "fedcorr", "feda3i"]
NICE = {"proposed": "Proposed (ours)", "feddyn": "FedDyn", "fednoro": "FedNoRo",
        "fedcorr": "FedCorr", "feda3i": "FedA3I"}


def load(ds, tier):
    p = SRC / f"{ds}_tier{tier}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("failed"):
                out.append(r)
    return out


def write(name, recs, cols=None):
    if not recs:
        return
    cols = cols or list(recs[0])
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    with open(OUT / "tables" / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(recs)
    with open(OUT / "tables" / f"{name}.md", "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in recs:
            f.write("| " + " | ".join("" if r.get(c) is None else str(r.get(c))
                                      for c in cols) + " |\n")


def weight_ratio(rec, key):
    """Mean weight of a ground-truth group relative to clean clients."""
    grp = set(rec.get(key) or [])
    others = set()
    for k in ("noisy_client_ids", "shifted_client_ids", "freerider_client_ids"):
        if k != key:
            others |= set(rec.get(k) or [])
    if not grp:
        return None
    rs = []
    for rd in rec.get("weights_by_round") or []:
        a, w = rd.get("active"), rd.get("weights")
        if not a or not w or len(a) != len(w):
            continue
        w = np.asarray(w, float)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        g = np.array([c in grp for c in a])
        c = np.array([(c not in grp) and (c not in others) for c in a])
        if g.sum() and c.sum():
            rs.append(w[g].mean() / max(w[c].mean(), 1e-12))
    return float(np.mean(rs)) if rs else None


def r3(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def main():
    OUT.mkdir(exist_ok=True)
    made = []

    # ---- A.3 / B.2 : corruption level x method, paired ------------------
    for ds in ("aptos", "isic", "octmnist"):
        rows = load(ds, 1)
        if not rows:
            continue
        acc = defaultdict(dict)
        for r in rows:
            acc[(r["overrides"]["noisy_client_frac"], r["seed"])][r["method"]] = r["acc"]
        lv = sorted({k[0] for k in acc})
        seeds = sorted({k[1] for k in acc})
        recs = []
        for m in METHODS:
            rec = {"method": NICE[m]}
            for l in lv:
                v = [acc[(l, s)][m] for s in seeds if m in acc.get((l, s), {})]
                rec[f"acc@{int(l*100)}%"] = r3(np.mean(v)) if v else None
                rec[f"n@{int(l*100)}%"] = len(v)
                if m != "feddyn":
                    d = [acc[(l, s)][m] - acc[(l, s)]["feddyn"] for s in seeds
                         if m in acc.get((l, s), {}) and "feddyn" in acc.get((l, s), {})]
                    rec[f"paired_delta@{int(l*100)}%"] = r3(np.mean(d)) if d else None
                    rec[f"sign@{int(l*100)}%"] = ("" if not d else
                        "all+" if all(x > 0 for x in d) else
                        "all-" if all(x < 0 for x in d) else "mixed")
            recs.append(rec)
        write(f"A3_B2_{ds}_corruption_level_x_method", recs)
        made.append(f"A3_B2_{ds}_corruption_level_x_method")

    # ---- A.3 : mechanism, who gets down-weighted ------------------------
    for ds in ("aptos", "isic", "octmnist"):
        rows = [r for t in (1, 2, 3) for r in load(ds, t)]
        recs = []
        for r in rows:
            c = weight_ratio(r, "noisy_client_ids")
            s = weight_ratio(r, "shifted_client_ids")
            p = weight_ratio(r, "freerider_client_ids")
            if c is None and s is None and p is None:
                continue
            recs.append({"axis": r["axis"], "method": NICE.get(r["method"], r["method"]),
                         "seed": r["seed"],
                         "corruption": r["overrides"].get("noisy_client_frac"),
                         "weight_ratio_corrupted": r3(c),
                         "weight_ratio_shifted": r3(s),
                         "weight_ratio_poisoner": r3(p),
                         "accuracy": r3(r.get("acc"))})
        if recs:
            write(f"A3_{ds}_mechanism_weight_ratios", recs)
            made.append(f"A3_{ds}_mechanism_weight_ratios")

    # ---- A.2 : corrupted vs shifted-but-correct -------------------------
    for ds in ("aptos", "isic", "octmnist"):
        rows = [r for r in load(ds, 2) if r["axis"] in ("mixed", "shift")]
        recs = []
        for r in rows:
            c = weight_ratio(r, "noisy_client_ids")
            s = weight_ratio(r, "shifted_client_ids")
            recs.append({"condition": r["axis"], "method": NICE.get(r["method"], r["method"]),
                         "weight_ratio_corrupted": r3(c),
                         "weight_ratio_shifted_but_CORRECT": r3(s),
                         "separation (shifted - corrupted)":
                             r3((s - c) if (c is not None and s is not None) else None),
                         "accuracy": r3(r.get("acc"))})
        if recs:
            write(f"A2_{ds}_corrupted_vs_shifted", recs)
            made.append(f"A2_{ds}_corrupted_vs_shifted")

    # ---- B.2/B.3/A.3/B.5 : the remaining axes ---------------------------
    LABEL = {"type": "B2_noise_type", "poison": "B3_model_poisoning",
             "alpha": "A3_noniid_degree", "part": "A3_participation_ratio",
             "scale": "B5_client_count"}
    for ds in ("aptos", "isic", "octmnist"):
        rows = [r for t in (2, 3) for r in load(ds, t)]
        for axis, lab in LABEL.items():
            sub = [r for r in rows if r["axis"] == axis]
            if not sub:
                continue
            recs = [{"method": NICE.get(r["method"], r["method"]),
                     **{k: v for k, v in r["overrides"].items()
                        if k in ("label_noise_type", "freerider_mode", "alpha",
                                 "active_size_m", "num_clients", "noisy_client_frac")},
                     "accuracy": r3(r.get("acc"))} for r in sub]
            write(f"{lab}_{ds}", recs)
            made.append(f"{lab}_{ds}")

    # ---- raw ------------------------------------------------------------
    (OUT / "raw").mkdir(exist_ok=True)
    for p in SRC.glob("*.jsonl"):
        shutil.copyfile(p, OUT / "raw" / p.name)

    (OUT / "README.md").write_text(
        "# Revision evidence, organised by reviewer comment\n\n"
        "Each table is named for the review point it answers.\n\n"
        "| file prefix | reviewer comment |\n|:--|:--|\n"
        "| `A2_` | A: how is corrupted data distinguished from difficult / shifted but correctly labelled data? |\n"
        "| `A3_` | A: noise levels, non-IID degrees, participation ratios, mechanism results on all datasets |\n"
        "| `B2_` | B: corruption levels and noise types |\n"
        "| `B3_` | B: model poisoning |\n"
        "| `B5_` | B: scaling beyond 10 clients |\n\n"
        "## How to read the main table\n\n"
        "`paired_delta` is the per-seed difference against FedDyn, averaged. Both\n"
        "methods see the identical partition and identical corrupted clients in a\n"
        "given seed, so the paired difference is the correct statistic: its spread\n"
        "is ~0.007 against ~0.045 for either arm alone. `sign` reports whether every\n"
        "seed agreed; `all+` with a positive delta is the claim, `mixed` is not.\n\n"
        "`weight_ratio_*` is the mean aggregation weight a ground-truth group\n"
        "received relative to clean clients, computed within each round and then\n"
        "averaged. Below 1 means down-weighted; 1.0 means treated identically.\n\n"
        "## Caveats carried from the runs\n\n"
        "- The three noise-aware baselines are mechanism-faithful reimplementations,\n"
        "  not ports; see `federated/noise_baselines.py` for exactly what differs.\n"
        "- OCTMNIST is subsampled to 20,000 training images; at full size one\n"
        "  40-round run costs 6.3 hours.\n"
        "- `n@` columns give the seed count behind each mean.\n")
    print(f"wrote {len(made)} tables -> {OUT.relative_to(ROOT)}/")
    for m in made:
        print("   ", m)


if __name__ == "__main__":
    main()

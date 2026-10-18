#!/usr/bin/env python3
"""Figures for the revision response."""
from __future__ import annotations
import json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "studies", "review")
OUT = os.path.join(ROOT, "figures", "review")
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
C = {"proposed": "#7b3fa0", "feddyn": "#0a7d3f", "fednoro": "#2a78d6",
     "fedcorr": "#eb6834", "feda3i": "#1baf7a"}
NICE = {"proposed": "Proposed (ours)", "feddyn": "FedDyn", "fednoro": "FedNoRo",
        "fedcorr": "FedCorr", "feda3i": "FedA3I"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.linewidth": .8, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight"})


def load(ds, tier):
    p = os.path.join(SRC, f"{ds}_tier{tier}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()
            and not json.loads(l).get("failed")]


# ---- fig 1: corruption level x method, all datasets --------------------
DS = [("aptos", "APTOS"), ("isic", "ISIC"), ("octmnist", "OCTMNIST (10k)")]
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), sharey=False)
for ax, (ds, title) in zip(axes, DS):
    rows = load(ds, 1)
    if not rows:
        ax.axis("off"); continue
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["method"]][r["overrides"]["noisy_client_frac"]].append(r["acc"])
    lv = sorted({l for m in acc for l in acc[m]})
    for m in ["feddyn", "fednoro", "fedcorr", "feda3i", "proposed"]:
        if m not in acc: continue
        y = [np.mean(acc[m][l]) if acc[m].get(l) else np.nan for l in lv]
        ax.plot([l*100 for l in lv], y, "o-", color=C[m], label=NICE[m],
                lw=2.4 if m == "proposed" else 1.3,
                ms=7 if m == "proposed" else 5,
                zorder=5 if m == "proposed" else 2)
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.set_xlabel("% clients corrupted"); ax.grid(color=GRID, lw=.6)
    ax.set_axisbelow(True); ax.set_xticks([l*100 for l in lv])
axes[0].set_ylabel("test accuracy")
axes[0].legend(fontsize=7.6, frameon=False)
fig.suptitle("Accuracy against corruption level — the submission tested only one point (40%)",
             fontsize=11.5, y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "rev_fig1_corruption_sweep.png"))
plt.close(fig)

# ---- fig 2: paired delta vs FedDyn -------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 3.8))
W = 0.26
for i, (ds, title) in enumerate(DS):
    rows = load(ds, 1)
    if not rows: continue
    by = defaultdict(dict)
    for r in rows:
        by[(r["overrides"]["noisy_client_frac"], r["seed"])][r["method"]] = r["acc"]
    lv = sorted({k[0] for k in by})
    d = []
    for l in lv:
        v = [by[(l, s)]["proposed"] - by[(l, s)]["feddyn"] for s in
             {k[1] for k in by if k[0] == l}
             if "proposed" in by[(l, s)] and "feddyn" in by[(l, s)]]
        d.append(np.mean(v) if v else np.nan)
    ax.bar(np.arange(len(lv)) + (i-1)*W, d, W, label=title,
           color=["#7b3fa0", "#2a78d6", "#eb6834"][i])
    for j, v in enumerate(d):
        if np.isfinite(v):
            ax.text(j + (i-1)*W, v + (0.0012 if v >= 0 else -0.0012),
                    f"{v:+.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=6.6, color=INK2)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(range(len(lv))); ax.set_xticklabels([f"{int(l*100)}%" for l in lv])
ax.margins(y=0.18)
ax.set_xlabel("% clients corrupted")
ax.set_ylabel("paired Δ accuracy vs FedDyn")
ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", color=GRID, lw=.6)
ax.set_axisbelow(True)
ax.set_title("Benefit appears at high corruption on two of three datasets",
             fontsize=10.5, color=INK)
fig.text(0.5, -0.07, "Paired per-seed difference: both methods see the identical partition and "
         "identical corrupted clients, so this is the correct statistic\n(paired spread ~0.007 "
         "against ~0.045 for either arm alone). ISIC is the counter-example and is reported as such.",
         ha="center", fontsize=7.6, color=INK2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "rev_fig2_paired_delta.png"))
plt.close(fig)
print("wrote rev_fig1_corruption_sweep.png, rev_fig2_paired_delta.png")

"""Cross-dataset figures for the characterization study (OCTMNIST/ISIC/APTOS).

Reads the frozen-protocol sweep logs for all three datasets, aggregates over
seeds, and writes:
  - crossdataset_acc_clean / crossdataset_acc_noise : grouped bars (3 ds x 3 methods)
  - opening_scatter : FedDyn noise-induced drop vs (proposed - FedDyn) gap
  - delta_heatmap   : proposed - FedDyn delta per dataset x metric x condition
plus results/crossdataset_tables.md.  Run: python3 scripts/make_crossdataset_figures.py
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs"); FIG = os.path.join(ROOT, "figures", "paper"); RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

DATASETS = ["octmnist", "isic", "aptos"]; DLABEL = {"octmnist": "OCTMNIST", "isic": "ISIC", "aptos": "APTOS"}
METHODS = ["fedavg", "feddyn", "proposed"]; MLABEL = {"fedavg": "FedAvg", "feddyn": "FedDyn", "proposed": "Proposed"}
COLOR = {"fedavg": "#0072B2", "feddyn": "#E69F00", "proposed": "#009E73"}
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 11, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "legend.frameon": False, "figure.autolayout": True})

def complete(f):
    e = [l for l in open(f) if '"type": "eval"' in l]
    return e and json.loads(e[-1]).get("round") == 39

def agg(ds, n, m, key="acc"):
    V = []
    for s in range(8):
        g = glob.glob(f"{LOGS}/{ds}_{m}_n{n}_seed{s}-*.jsonl")
        if g and complete(g[0]):
            e = [json.loads(l) for l in open(g[0]) if '"type": "eval"' in l][-1]
            if key in e and e[key] is not None: V.append(e[key])
    return (np.mean(V), np.std(V), len(V)) if V else (np.nan, 0, 0)

def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(f"{FIG}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig); print("wrote", name)

# ---- grouped bars: accuracy per dataset, one figure per condition ----
def bars(n, lab, fname):
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); x = np.arange(len(DATASETS)); w = 0.26
    for i, m in enumerate(METHODS):
        mu = [agg(d, n, m)[0] for d in DATASETS]; sd = [agg(d, n, m)[1] for d in DATASETS]
        b = ax.bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=3, color=COLOR[m], label=MLABEL[m],
                   edgecolor="white", linewidth=0.8)
        for bb, v in zip(b, mu):
            if not np.isnan(v): ax.text(bb.get_x() + bb.get_width() / 2, v + 0.015, f"{v:.2f}",
                                        ha="center", fontsize=8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([DLABEL[d] for d in DATASETS]); ax.set_ylim(0, 0.95)
    ax.set_ylabel("Accuracy"); ax.set_title(f"Accuracy across datasets — {lab}", fontsize=11.5)
    ax.legend(ncol=3, loc="upper right", fontsize=9.5); save(fig, fname)

bars("00", "clean (0% noise)", "crossdataset_acc_clean")
bars("40", "40% label noise", "crossdataset_acc_noise")

# ---- the KEY figure: opening (FedDyn drop) vs proposed-FedDyn gap at 40% ----
fig, ax = plt.subplots(figsize=(7.0, 5.0))
for d in DATASETS:
    fd_clean = agg(d, "00", "feddyn")[0]; fd_noise = agg(d, "40", "feddyn")[0]
    pr_noise = agg(d, "40", "proposed")[0]
    opening = fd_clean - fd_noise           # how much noise hurts FedDyn
    gap = pr_noise - fd_noise               # proposed advantage over FedDyn under noise
    ax.scatter(opening, gap, s=140, color="#009E73", zorder=3, edgecolor="white", linewidth=1)
    ax.annotate(f"{DLABEL[d]}", (opening, gap), textcoords="offset points", xytext=(10, 6),
                fontsize=10.5, fontweight="bold")
ax.axhline(0, color="#888", lw=1.2, ls="--")
ax.text(0.02, 0.006, "proposed better than FedDyn ↑", fontsize=8.5, color="#555", transform=ax.get_yaxis_transform() if False else ax.transData)
ax.set_xlabel("FedDyn accuracy drop from noise  (clean − 40%)  →  bigger opening")
ax.set_ylabel("Proposed − FedDyn accuracy at 40% noise")
ax.set_title("When does down-weighting help?\nProposed only approaches FedDyn where noise hurts FedDyn most", fontsize=11)
save(fig, "opening_scatter")

# ---- delta heatmap: proposed - feddyn, per dataset x (metric,condition) ----
cells = [("acc", "00", "Acc·clean"), ("acc", "40", "Acc·40%"),
         ("jain", "40", "Jain·40%"), ("client_acc_min", "40", "Worst·40%")]
M = np.array([[agg(d, n, "proposed", k)[0] - agg(d, n, "feddyn", k)[0] for (k, n, _) in cells] for d in DATASETS])
fig, ax = plt.subplots(figsize=(7.2, 3.6))
vmax = np.nanmax(np.abs(M)); im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(cells))); ax.set_xticklabels([c[2] for c in cells])
ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels([DLABEL[d] for d in DATASETS])
for i in range(len(DATASETS)):
    for j in range(len(cells)):
        ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=10,
                color="#000")
ax.set_title("Proposed − FedDyn (green = proposed better, red = worse)", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); save(fig, "delta_heatmap")

# ---- table ----
lines = ["# Cross-dataset results (3 seeds, last round; APTOS also reports QWK)\n"]
for n, lab in [("00", "Clean (0% noise)"), ("40", "40% label noise")]:
    lines.append(f"\n### {lab}\n")
    lines.append("| Dataset | Method | Accuracy | Jain | Worst-client |")
    lines.append("|---|---|---|---|---|")
    for d in DATASETS:
        for m in METHODS:
            a = agg(d, n, m); j = agg(d, n, m, "jain"); w = agg(d, n, m, "client_acc_min")
            lines.append(f"| {DLABEL[d]} | {MLABEL[m]} | {a[0]:.3f}±{a[1]:.3f} (n={a[2]}) | {j[0]:.3f} | {w[0]:.3f} |")
open(f"{RES}/crossdataset_tables.md", "w").write("\n".join(lines) + "\n")
print("wrote results/crossdataset_tables.md")

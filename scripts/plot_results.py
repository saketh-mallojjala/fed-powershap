"""Generate publication-style figures from one or more JSONL logs.

Usage:
    python scripts/plot_results.py                          # plots newest log
    python scripts/plot_results.py logs/run1.jsonl          # single run
    python scripts/plot_results.py logs/a.jsonl logs/b.jsonl logs/c.jsonl
                                                            # overlay comparison

Outputs go to `figures/<exp_stem>/` as PNG + PDF.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# ---------- style ----------

PALETTE = ["#2E86AB", "#E63946", "#06A77D", "#F4A261", "#8E44AD", "#2A9D8F"]
CSSV_CMAP = LinearSegmentedColormap.from_list(
    "cssv", ["#1b1f3b", "#2E86AB", "#f1faee", "#F4A261", "#E63946"]
)
SELECTION_CMAP = LinearSegmentedColormap.from_list(
    "sel", ["#f7f7f7", "#2E86AB", "#1b1f3b"]
)

CIFAR10_CLASSES = ["plane", "auto", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
MNIST_CLASSES = [str(i) for i in range(10)]
APTOS_CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.family": "DejaVu Sans",
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


# ---------- data ----------

@dataclass
class Run:
    label: str
    cfg: Dict
    sizes: List[int]
    class_hists: List[List[int]]
    rounds: List[Dict]
    evals: List[Dict]

    @property
    def num_classes(self) -> int:
        return int(self.cfg.get("num_classes", 10))

    @property
    def num_clients(self) -> int:
        return int(self.cfg.get("num_clients", len(self.sizes)))


def load_run(path: str) -> Run:
    cfg, sizes, hists, rounds, evals = {}, [], [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            t = r.get("type")
            if t == "config":
                cfg = r
            elif t == "client_sizes":
                sizes = r["sizes"]
                hists = r.get("class_histograms", [])
            elif t == "round":
                rounds.append(r)
            elif t == "eval":
                evals.append(r)
    label = f"{cfg.get('selection_strategy','?')}+{cfg.get('aggregation','?')}"
    return Run(label=label, cfg=cfg, sizes=sizes, class_hists=hists,
               rounds=rounds, evals=evals)


def class_names(run: Run) -> List[str]:
    if run.cfg.get("dataset") == "cifar10":
        return CIFAR10_CLASSES
    if run.cfg.get("dataset") == "aptos":
        return APTOS_CLASSES
    return MNIST_CLASSES[: run.num_classes]


# ---------- plots ----------

def _smooth(y, w=5):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, k, mode="valid")[: len(y)]


def plot_accuracy(runs: List[Run], out: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, run in enumerate(runs):
        if not run.evals:
            continue
        x = [e["round"] for e in run.evals]
        y = [e["acc"] for e in run.evals]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, color=color, alpha=0.25, lw=1.2)
        ax.plot(x, _smooth(y), color=color, lw=2.3, label=run.label)
        best = max(run.evals, key=lambda e: e["acc"])
        ax.scatter([best["round"]], [best["acc"]], color=color, s=80,
                   zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{best['acc']:.3f}",
                    (best["round"], best["acc"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=color, fontweight="bold")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy over rounds")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def plot_qwk(runs: List[Run], out: str):
    """Quadratic Weighted Kappa over rounds (APTOS metric)."""
    any_qwk = any("qwk" in e for r in runs for e in r.evals)
    if not any_qwk:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, run in enumerate(runs):
        evals = [e for e in run.evals if "qwk" in e]
        if not evals:
            continue
        x = [e["round"] for e in evals]
        y = [e["qwk"] for e in evals]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, color=color, alpha=0.25, lw=1.2)
        ax.plot(x, _smooth(y), color=color, lw=2.3, label=run.label)
        best = max(evals, key=lambda e: e["qwk"])
        ax.scatter([best["round"]], [best["qwk"]], color=color, s=80,
                   zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{best['qwk']:.3f}",
                    (best["round"], best["qwk"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=color, fontweight="bold")
    ax.axhline(0.0, ls=":", color="#888", lw=1)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Quadratic Weighted Kappa")
    ax.set_title("QWK over rounds (APTOS official metric)")
    ax.set_ylim(-0.1, 1.0)
    ax.legend(loc="lower right")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def plot_loss(runs: List[Run], out: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, run in enumerate(runs):
        if not run.evals:
            continue
        x = [e["round"] for e in run.evals]
        y = [e["loss"] for e in run.evals]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, color=color, alpha=0.25, lw=1.2)
        ax.plot(x, _smooth(y), color=color, lw=2.3, label=run.label)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test loss (cross-entropy)")
    ax.set_title("Test loss over rounds")
    ax.legend(loc="upper right")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def plot_per_class_final(run: Run, out: str):
    if not run.evals:
        return
    final = run.evals[-1]
    per_class = np.array(final["per_class_acc"])
    names = class_names(run)
    order = np.argsort(per_class)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(per_class[order])
    bars = ax.bar(range(len(per_class)), per_class[order],
                  color=colors, edgecolor="#333", linewidth=0.7)
    for b, v in zip(bars, per_class[order]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", fontsize=9, color="#222")
    ax.set_xticks(range(len(per_class)))
    ax.set_xticklabels([names[i] for i in order], rotation=20)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Per-class accuracy at round {final['round']} — {run.label}")
    ax.axhline(np.mean(per_class), ls="--", color="#333", lw=1,
               label=f"mean = {np.mean(per_class):.3f}")
    ax.legend(loc="upper right")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def plot_selection_frequency(run: Run, out: str):
    counts = Counter()
    for r in run.rounds:
        counts.update(r["active"])
    ids = list(range(run.num_clients))
    freq = np.array([counts.get(i, 0) for i in ids])
    sizes = np.array(run.sizes)

    fig, ax = plt.subplots(figsize=(10, 5))
    norm_size = (sizes - sizes.min()) / max(1, np.ptp(sizes))
    colors = plt.cm.viridis(norm_size)
    bars = ax.bar(ids, freq, color=colors, edgecolor="#333", linewidth=0.6)
    for b, v in zip(bars, freq):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v),
                ha="center", fontsize=8, color="#222")
    ax.set_xlabel("Client id")
    ax.set_ylabel(f"# rounds trained (of {len(run.rounds)})")
    ax.set_title(f"Client selection frequency — {run.label}\n"
                 "(bar color = dataset size; taller = selected more)")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(sizes.min(), sizes.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Dataset size")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def plot_selection_heatmap(run: Run, out: str):
    """Client × round heatmap: cell = 1 if client was in active set that round."""
    N = run.num_clients
    T = len(run.rounds)
    mat = np.zeros((N, T))
    for r in run.rounds:
        for cid in r["active"]:
            mat[cid, r["round"]] = 1.0

    fig, ax = plt.subplots(figsize=(min(14, 1 + 0.12 * T), 0.3 * N + 1.5))
    ax.imshow(mat, aspect="auto", cmap=SELECTION_CMAP, interpolation="nearest")
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"c{i}" for i in range(N)], fontsize=8)
    ax.set_xlabel("Round")
    ax.set_title(f"Active clients per round — {run.label}")
    ax.grid(False)
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def plot_cssv_heatmap(run: Run, out: str):
    """Average CSSV matrix across rounds: client × class."""
    if run.cfg.get("aggregation") != "shapfed_wa":
        return
    N, C = run.num_clients, run.num_classes
    agg = np.zeros((N, C))
    seen = np.zeros(N)
    for r in run.rounds:
        cssv = r.get("cssv")
        if not cssv:
            continue
        for local_i, cid in enumerate(r["active"]):
            agg[cid] += np.array(cssv[local_i])
            seen[cid] += 1
    seen[seen == 0] = 1
    avg = agg / seen[:, None]

    fig, ax = plt.subplots(figsize=(7, 0.35 * N + 1.5))
    vmax = max(abs(avg.min()), abs(avg.max()), 1e-6)
    im = ax.imshow(avg, cmap=CSSV_CMAP, aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(C))
    ax.set_xticklabels(class_names(run), rotation=30, fontsize=9)
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"c{i}" for i in range(N)], fontsize=8)
    ax.set_title(f"Mean CSSV φ(k, c) across rounds — {run.label}\n"
                 "(warm = positive contribution, cool = opposing)")
    fig.colorbar(im, ax=ax, pad=0.02).set_label("CSSV (cosine sim)")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def plot_weight_timeline(run: Run, out: str):
    """Stack plot of aggregation weights over time, per client."""
    N = run.num_clients
    T = len(run.rounds)
    mat = np.zeros((N, T))
    for r in run.rounds:
        for local_i, cid in enumerate(r["active"]):
            mat[cid, r["round"]] = r["weights"][local_i]

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.cm.tab20
    bottoms = np.zeros(T)
    for cid in range(N):
        ax.fill_between(range(T), bottoms, bottoms + mat[cid],
                        color=cmap(cid % 20), alpha=0.9,
                        label=f"c{cid}" if N <= 20 else None,
                        linewidth=0)
        bottoms += mat[cid]
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1.001)
    ax.set_xlabel("Round")
    ax.set_ylabel("Aggregation weight share")
    ax.set_title(f"Per-client aggregation weights over time — {run.label}")
    if N <= 20:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  ncol=1, fontsize=8)
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def plot_client_data_distribution(run: Run, out: str):
    if not run.class_hists:
        return
    hists = np.array(run.class_hists)  # (N, C)
    fig, ax = plt.subplots(figsize=(11, 0.35 * run.num_clients + 1.5))
    im = ax.imshow(hists, aspect="auto", cmap="magma")
    ax.set_xticks(range(run.num_classes))
    ax.set_xticklabels(class_names(run), rotation=30, fontsize=9)
    ax.set_yticks(range(run.num_clients))
    ax.set_yticklabels([f"c{i}" for i in range(run.num_clients)], fontsize=8)
    ax.set_title(f"Per-client class distribution (Dirichlet α={run.cfg.get('alpha')})")
    fig.colorbar(im, ax=ax, pad=0.02).set_label("# samples")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def plot_summary_card(run: Run, out: str):
    """A single 2x2 dashboard summarizing the run."""
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # (0,0) accuracy
    ax = fig.add_subplot(gs[0, 0])
    if run.evals:
        x = [e["round"] for e in run.evals]
        y = [e["acc"] for e in run.evals]
        ax.plot(x, y, color=PALETTE[0], alpha=0.3, lw=1)
        ax.plot(x, _smooth(y), color=PALETTE[0], lw=2.5)
        best = max(run.evals, key=lambda e: e["acc"])
        ax.scatter([best["round"]], [best["acc"]], color=PALETTE[1],
                   s=90, zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate(f"best {best['acc']:.3f}",
                    (best["round"], best["acc"]),
                    textcoords="offset points", xytext=(8, -4),
                    fontsize=10, color=PALETTE[1], fontweight="bold")
    ax.set_title("Accuracy")
    ax.set_xlabel("Round"); ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)

    # (0,1) loss
    ax = fig.add_subplot(gs[0, 1])
    if run.evals:
        x = [e["round"] for e in run.evals]
        y = [e["loss"] for e in run.evals]
        ax.plot(x, y, color=PALETTE[2], alpha=0.3, lw=1)
        ax.plot(x, _smooth(y), color=PALETTE[2], lw=2.5)
    ax.set_title("Test loss"); ax.set_xlabel("Round"); ax.set_ylabel("CE loss")

    # (1,0) per-class final
    ax = fig.add_subplot(gs[1, 0])
    if run.evals:
        per_class = np.array(run.evals[-1]["per_class_acc"])
        names = class_names(run)
        order = np.argsort(per_class)[::-1]
        colors = plt.cm.RdYlGn(per_class[order])
        ax.bar(range(len(per_class)), per_class[order],
               color=colors, edgecolor="#333", linewidth=0.6)
        ax.set_xticks(range(len(per_class)))
        ax.set_xticklabels([names[i] for i in order], rotation=25, fontsize=9)
        ax.axhline(np.mean(per_class), ls="--", color="#333", lw=1)
        ax.set_ylim(0, 1.1)
    ax.set_title("Per-class accuracy (final round)")
    ax.set_ylabel("Accuracy")

    # (1,1) selection frequency
    ax = fig.add_subplot(gs[1, 1])
    counts = Counter()
    for r in run.rounds:
        counts.update(r["active"])
    ids = list(range(run.num_clients))
    freq = [counts.get(i, 0) for i in ids]
    sizes = np.array(run.sizes)
    norm = (sizes - sizes.min()) / max(1, np.ptp(sizes))
    ax.bar(ids, freq, color=plt.cm.viridis(norm),
           edgecolor="#333", linewidth=0.5)
    ax.set_title("Client selection frequency")
    ax.set_xlabel("Client id")
    ax.set_ylabel("# rounds trained")

    qwk_str = ""
    if run.evals:
        best_qwk_eval = max((e for e in run.evals if "qwk" in e),
                            key=lambda e: e["qwk"], default=None)
        if best_qwk_eval is not None:
            qwk_str = f"  •  best QWK={best_qwk_eval['qwk']:.3f}"
    fig.suptitle(
        f"{run.label}  •  {run.cfg.get('dataset')}  •  N={run.num_clients}  "
        f"•  m={run.cfg.get('active_size_m')}/d={run.cfg.get('candidate_size_d')}  "
        f"•  α={run.cfg.get('alpha')}  •  {len(run.rounds)} rounds{qwk_str}",
        fontsize=14, fontweight="bold", y=1.00,
    )
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


# ---------- driver ----------

def newest_log(logs_dir: str = "logs") -> Optional[str]:
    files = sorted(glob.glob(os.path.join(logs_dir, "*.jsonl")),
                   key=os.path.getmtime)
    return files[-1] if files else None


def main(argv):
    set_style()
    paths = argv[1:] if len(argv) > 1 else []
    if not paths:
        p = newest_log()
        if not p:
            print("No logs/*.jsonl found. Run main.py first."); sys.exit(1)
        paths = [p]
    runs = [load_run(p) for p in paths]

    stem = "+".join(os.path.splitext(os.path.basename(p))[0] for p in paths)
    outdir = os.path.join("figures", stem[:80])
    os.makedirs(outdir, exist_ok=True)

    # Comparison plots (work for 1+ runs).
    plot_accuracy(runs, os.path.join(outdir, "01_accuracy"))
    plot_loss(runs, os.path.join(outdir, "02_loss"))
    plot_qwk(runs, os.path.join(outdir, "02b_qwk"))

    # Per-run plots.
    for run in runs:
        tag = run.label.replace("+", "-")
        plot_summary_card(run, os.path.join(outdir, f"00_summary_{tag}"))
        plot_per_class_final(run, os.path.join(outdir, f"03_per_class_{tag}"))
        plot_selection_frequency(run, os.path.join(outdir, f"04_selection_freq_{tag}"))
        plot_selection_heatmap(run, os.path.join(outdir, f"05_selection_heatmap_{tag}"))
        plot_cssv_heatmap(run, os.path.join(outdir, f"06_cssv_heatmap_{tag}"))
        plot_weight_timeline(run, os.path.join(outdir, f"07_weight_timeline_{tag}"))
        plot_client_data_distribution(run, os.path.join(outdir, f"08_data_dist_{tag}"))

    print(f"Wrote figures to {outdir}/")
    for f in sorted(os.listdir(outdir)):
        if f.endswith(".png"):
            print(f"  {os.path.join(outdir, f)}")


if __name__ == "__main__":
    main(sys.argv)

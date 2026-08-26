#!/usr/bin/env python3
"""Figures for the DCFI-UPA final report.

Colours are the dataviz reference palette's categorical slots, taken in fixed
order (blue, orange, aqua, yellow) rather than cycled. Marks are thin, grids
recessive, and every multi-series figure carries a legend, so identity is never
carried by colour alone.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures", "dcfi")
os.makedirs(OUT, exist_ok=True)

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # validated slots
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8b8a85"
GRID = "#e3e2de"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _clean(ax, xgrid=False, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.7)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.7)


# ---------------------------------------------------------------- figure 1
def fig_baselines():
    """Ranked comparison under attack. Horizontal bars: the job is magnitude
    with a long category axis, so rank reads top-to-bottom without rotated
    labels. DCFI is emphasised; every other method shares one recessive hue,
    because the comparison is 'ours vs the field', not eight identities."""
    data = [("DCFI (ours)", 0.776, 0.020), ("FedDyn", 0.770, 0.021),
            ("FedCE", 0.761, 0.038), ("FedBN", 0.700, 0.115),
            ("FedProx", 0.700, 0.116), ("FedAvg", 0.700, 0.116),
            ("MOON", 0.678, 0.139), ("SCAFFOLD", 0.524, 0.040)]
    names = [d[0] for d in data][::-1]
    vals = np.array([d[1] for d in data])[::-1]
    errs = np.array([d[2] for d in data])[::-1]
    cols = [C1 if n.startswith("DCFI") else "#c9d4e2" for n in names]

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.62, color=cols, zorder=3)
    ax.errorbar(vals, y, xerr=errs, fmt="none", ecolor=INK2, elinewidth=1.1,
                capsize=2.5, zorder=4)
    ax.set_yticks(y, names)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0.45, 0.86)
    ax.set_xlabel("Balanced accuracy (higher is better)")
    _clean(ax, xgrid=True, ygrid=False)
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.008, yi, f"{v:.3f}", va="center", fontsize=8.3,
                color=INK, fontweight="bold" if yi == len(y) - 1 else "normal")
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(0.503, -0.62, "majority-class floor", fontsize=7.4, color=MUTED)
    ax.set_title("Under attack (zero-update free-riders, 40% of clients)",
                 fontsize=10, color=INK, loc="left", pad=8)
    fig.savefig(f"{OUT}/fig1_baselines.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_alpha():
    """Two series over an ordered x — a line chart. Log x because alpha spans
    two orders of magnitude. Both panels share one y-scale so the panels are
    comparable at a glance (never two y-axes on one plot)."""
    a = [0.1, 0.3, 0.5, 1.0, 10.0]
    fed = {"none": [0.750, 0.773, 0.778, 0.785, 0.786],
           "zero_update": [0.708, 0.749, 0.764, 0.767, 0.779]}
    dcf = {"none": [0.726, 0.760, 0.783, 0.784, 0.788],
           "zero_update": [0.686, 0.770, 0.775, 0.755, 0.784]}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, key, title in zip(axes, ["none", "zero_update"],
                              ["No adversaries", "40% zero-update free-riders"]):
        ax.plot(a, fed[key], "-o", color=C2, lw=2, ms=5.5, label="FedDyn", zorder=3)
        ax.plot(a, dcf[key], "-o", color=C1, lw=2, ms=5.5, label="DCFI (ours)", zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(a, [str(x) for x in a])
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)
        _clean(ax)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    # One shared axis label: a per-panel label collides at the panel boundary.
    fig.supxlabel("Dirichlet α    (← more heterogeneous · more IID →)",
                  fontsize=9, color=INK2, y=-0.04)
    fig.savefig(f"{OUT}/fig2_alpha.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_capture():
    """Adversary budget capture per attack, against the fair-share reference.
    Sorted ascending so the two regimes separate visually; the 40% reference
    line is what any mechanism-free rule would pay them."""
    atk = [("zero_update", 0.00), ("copied_update", 3.33), ("scaled_global", 6.25),
           ("random_update", 9.12), ("stale_model", 31.75), ("low_effort", 33.53),
           ("norm_inflation", 37.10), ("label_flip", 37.38)]
    names = [a[0] for a in atk]
    vals = [a[1] for a in atk]
    cols = [C3 if v <= 10 else C2 for v in vals]

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(names))
    ax.bar(x, vals, width=0.6, color=cols, zorder=3)
    ax.axhline(40, color=INK2, lw=1.4, ls=(0, (4, 3)), zorder=4)
    ax.text(7.45, 41.2, "40% = their fair share\n(what equal-split pays)",
            fontsize=7.6, color=INK2, ha="right", va="bottom")
    ax.axhline(5, color=C3, lw=1, ls=(0, (2, 2)), zorder=4)
    ax.text(-0.42, 6.2, "5% target", fontsize=7.4, color=C3)
    ax.set_xticks(x, names, rotation=28, ha="right")
    ax.set_ylabel("Share of reward budget captured (%)")
    ax.set_ylim(0, 52)
    _clean(ax)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.1, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.legend(handles=[Patch(facecolor=C3, label="Starved — structurally impossible update"),
                       Patch(facecolor=C2, label="Not starved — performs genuine training")],
              frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.02))
    fig.savefig(f"{OUT}/fig3_capture.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_c1():
    """The C1 verdict. A dot-and-interval plot, not bars: the quantity is a
    correlation with uncertainty, and overlapping intervals ARE the finding —
    no rule separates from any other, and every interval crosses zero."""
    rules = ["fedshapin (ours)", "size-proportional", "equal-split",
             "raw CSSV", "quality-only"]
    rho = [0.086, 0.107, 0.097, 0.091, 0.095]
    lo = [-0.199, -0.094, -0.181, -0.183, -0.187]
    hi = [0.416, 0.391, 0.433, 0.408, 0.425]

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    y = np.arange(len(rules))[::-1]
    cols = [C1 if r.startswith("fedshapin") else MUTED for r in rules]
    for yi, r, l, h, c in zip(y, rho, lo, hi, cols):
        ax.plot([l, h], [yi, yi], color=c, lw=2.2, solid_capstyle="round", zorder=3)
        ax.plot([r], [yi], "o", color=c, ms=8, zorder=4,
                markeredgecolor="white", markeredgewidth=1.5)
    ax.axvline(0, color=INK2, lw=1.2, zorder=2)
    ax.axvline(0.15, color=C2, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.text(0.157, len(rules) - 0.35, "pre-registered bar\n(+0.15 over every baseline)",
            fontsize=7.6, color=C2, va="top")
    ax.set_yticks(y, rules)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel("Spearman ρ (reward vs independent leave-one-out contribution)")
    ax.set_xlim(-0.32, 0.62)
    _clean(ax, xgrid=True, ygrid=False)
    fig.savefig(f"{OUT}/fig4_c1.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_gate():
    """The design ablation that decided the architecture. Grouped bars: three
    configurations across four scenarios, one y-scale, legend always present."""
    sc = ["clean", "zero_update", "copied_update", "random_update"]
    fed = [0.768, 0.768, 0.777, 0.765]
    full = [0.735, 0.647, 0.643, 0.637]
    rew = [0.737, 0.752, 0.762, 0.750]

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(sc))
    w = 0.26
    ax.bar(x - w, fed, w - 0.02, label="FedDyn (no mechanism)", color="#c9d4e2", zorder=3)
    ax.bar(x, full, w - 0.02, label="Gate on aggregation", color=C2, zorder=3)
    ax.bar(x + w, rew, w - 0.02, label="Gate on reward only (ours)", color=C1, zorder=3)
    ax.set_xticks(x, sc)
    ax.set_ylim(0.55, 0.83)
    ax.set_ylabel("Balanced accuracy")
    _clean(ax)
    ax.legend(frameon=False, fontsize=8.3, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    for xi, v in zip(x, full):
        ax.text(xi, v - 0.018, f"{v:.3f}", ha="center", fontsize=7.6, color="white")
    for xi, v in zip(x, rew):
        ax.text(xi + w, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.6, color=INK)
    fig.savefig(f"{OUT}/fig5_gate.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_baselines, fig_alpha, fig_capture, fig_c1, fig_gate):
        f()
        print("wrote", f.__name__)
    print(f"-> {OUT}")

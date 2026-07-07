"""Flow diagram of the proposed federated round (one communication round).

Standard steps in blue; the novel robustness block (loss detector + loss-based
weighting + weight-consistent FedDyn aggregation) highlighted in green.
Writes figures/paper/architecture.{png,pdf}.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "paper")
os.makedirs(OUT, exist_ok=True)

BLUE = "#0072B2"; GREEN = "#009E73"; ORANGE = "#D55E00"
INK = "#1a1a1a"; MUTED = "#555"

plt.rcParams.update({"font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(8.6, 11))
ax.set_xlim(0, 10); ax.set_ylim(-1.4, 15.2); ax.axis("off")

def box(y, text, color, h=1.15, w=7.4, x=1.3, fc=None, fontsize=11, bold_first=False):
    fc = fc or (color + "18")
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.16",
                                linewidth=2, edgecolor=color, facecolor=fc, zorder=2))
    if bold_first:
        lines = text.split("\n", 1)
        ax.text(x + w/2, y + h*0.66, lines[0], ha="center", va="center",
                fontsize=fontsize+0.5, fontweight="bold", color=INK, zorder=3)
        if len(lines) > 1:
            ax.text(x + w/2, y + h*0.28, lines[1], ha="center", va="center",
                    fontsize=fontsize-1.5, color=MUTED, zorder=3)
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=INK, zorder=3)

def arrow(y1, y2, x=5.0, color=INK, label=None):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=18,
                                 linewidth=1.8, color=color, zorder=1))
    if label:
        ax.text(x + 0.25, (y1+y2)/2, label, ha="left", va="center", fontsize=8.5,
                color=MUTED, style="italic")

# ---- title ----
ax.text(5, 14.9, "Proposed federated round", ha="center", fontsize=15, fontweight="bold", color=INK)
ax.text(5, 14.5, "FedDyn optimizer  +  loss-based client-quality down-weighting",
        ha="center", fontsize=10.5, color=MUTED)

# ---- standard steps (blue) ----
box(13.0, "Global model  wₜ\n(server broadcasts to clients)", BLUE, bold_first=True)
arrow(13.0, 12.35)
box(11.2, "Client selection — random\npick m = 5 of N = 10 clients each round", BLUE, bold_first=True)
arrow(11.2, 10.55)
box(9.4, "Local training — FedDyn\ndynamic regularization limits client drift", BLUE, bold_first=True)
arrow(9.4, 8.75, label="client updates wₖ")

# ---- novel block (green) : big container ----
ax.add_patch(FancyBboxPatch((0.7, 2.55), 8.6, 6.05, boxstyle="round,pad=0.1,rounding_size=0.2",
                            linewidth=2.4, edgecolor=GREEN, facecolor="none",
                            linestyle=(0, (6, 3)), zorder=1))
ax.text(1.05, 8.35, "NOVEL: robustness mechanism", ha="left", va="center",
        fontsize=10, fontweight="bold", color=GREEN)

box(7.0, "Client-quality detector  (loss)\nlossₖ = L(wₜ ; client k's own data)", GREEN, bold_first=True)
# side annotation
ax.text(9.05, 7.57, "corrupted client\n→ high loss", ha="left", va="center",
        fontsize=8.5, color=ORANGE, fontweight="bold")
arrow(7.0, 6.35)
box(4.8, "Loss-based weights\nwₖ ∝ exp(−β·lossₖ),  blended with size prior,  capped ≤ 0.30",
    GREEN, bold_first=True, fontsize=10.5)
ax.text(9.05, 5.37, "high loss\n→ low weight\n(down-weighted)", ha="left", va="center",
        fontsize=8.5, color=ORANGE, fontweight="bold")
arrow(4.8, 4.15)
box(2.6, "Weight-consistent FedDyn aggregation\ndrift-correction uses the same weights → wₜ₊₁",
    GREEN, bold_first=True, fontsize=10.5)

# ---- loop back ----
arrow(2.6, 1.75, label="cosine LR decay")
box(0.55, "Updated global model  wₜ₊₁   →   next round", BLUE, bold_first=True, h=1.0)
# curved loop arrow on the left
ax.add_patch(FancyArrowPatch((1.3, 1.05), (0.35, 1.05), arrowstyle="-", linewidth=1.6, color=MUTED))
ax.add_patch(FancyArrowPatch((0.35, 1.05), (0.35, 13.57), arrowstyle="-", linewidth=1.6, color=MUTED))
ax.add_patch(FancyArrowPatch((0.35, 13.57), (1.3, 13.57), arrowstyle="-|>", mutation_scale=16,
                             linewidth=1.6, color=MUTED))
ax.text(0.15, 7.3, "repeat for 40 rounds", rotation=90, ha="center", va="center",
        fontsize=8.5, color=MUTED, style="italic")

# ---- caption ----
ax.text(5, -1.0,
        "Result: matches FedDyn (strongest baseline) — parity accuracy, lower variance,\n"
        "and a verified mechanism that down-weights corrupted clients (weight ratio 0.69).",
        ha="center", va="bottom", fontsize=9, color=MUTED)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"architecture.{ext}"), bbox_inches="tight", dpi=300)
print("wrote figures/paper/architecture.png/.pdf")

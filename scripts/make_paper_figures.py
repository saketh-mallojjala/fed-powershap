"""Publication-quality figures + tables for the robust-ShapFed noise study.

Reads the frozen-protocol sweep logs (octmnist_{method}_n{NN}_seed{s}.jsonl),
aggregates over seeds, and writes a full figure suite (PNG @300dpi + PDF) to
figures/paper/, plus paper-ready Markdown + LaTeX tables to results/.

Whatever noise levels are present get used, so this can be re-run as the sweep
fills in (0/20/40/60%). Run:  python3 scripts/make_paper_figures.py
"""
from __future__ import annotations
import glob, json, os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
FIGDIR = os.path.join(ROOT, "figures", "paper")
RESDIR = os.path.join(ROOT, "results")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)

METHODS = ["fedavg", "feddyn", "proposed"]
LABEL = {"fedavg": "FedAvg", "feddyn": "FedDyn", "proposed": "Proposed"}
# Okabe-Ito colourblind-safe categorical palette; Proposed = green (hero).
COLOR = {"fedavg": "#0072B2", "feddyn": "#E69F00", "proposed": "#009E73"}
MARKER = {"fedavg": "o", "feddyn": "s", "proposed": "^"}
LSTYLE = {"fedavg": "--", "feddyn": "-.", "proposed": "-"}
LW = {"fedavg": 2.0, "feddyn": 2.0, "proposed": 2.6}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "font.family": "sans-serif", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.autolayout": True,
})


def noisy_ids(num_clients, frac, seed):
    if frac <= 0:
        return set()
    rng = np.random.default_rng(seed + 2)  # matches data/dataset.py injection
    k = max(1, int(round(num_clients * frac)))
    return set(rng.choice(num_clients, size=k, replace=False).tolist())


def parse(path):
    cfg = None; rounds = []; evals = []
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        t = r.get("type")
        if t == "config": cfg = r
        elif t == "round": rounds.append(r)
        elif t == "eval": evals.append(r)
    return cfg, rounds, evals


# data[noise][method] = list of per-seed dicts {last:..., curve:{acc/jain/worst by round}, rounds, wr:{noisy,clean per round}}
data = defaultdict(lambda: defaultdict(list))
for path in glob.glob(os.path.join(LOGS, "octmnist_*_n*_seed*.jsonl")):
    base = os.path.basename(path)
    parts = base.split("-")[0].split("_")
    if len(parts) != 4:
        continue
    _, method, ntok, _ = parts
    if method not in METHODS or not re.fullmatch(r"n\d+", ntok):
        continue
    noise = int(ntok[1:])
    cfg, rounds, evals = parse(path)
    if not cfg or not evals or evals[-1].get("round") != cfg["num_rounds"] - 1:
        continue  # only fully-complete runs
    if cfg.get("num_rounds") != 40:
        continue  # restrict to the frozen 40-round protocol (excludes old runs)
    seed = cfg["seed"]; N = cfg["num_clients"]
    nz = noisy_ids(N, cfg.get("noisy_client_frac", 0.0), seed)
    curve = {"acc": {}, "jain": {}, "worst": {}}
    for e in evals:
        rd = e["round"]
        curve["acc"][rd] = e.get("acc")
        curve["jain"][rd] = e.get("jain")
        curve["worst"][rd] = e.get("client_acc_min")
    # per-round mean weight to noisy vs clean (proposed only, meaningful under noise)
    wr = {"noisy": {}, "clean": {}}
    for r in rounds:
        act = r.get("active") or []; w = r.get("weights") or []
        wn = [wi for cid, wi in zip(act, w) if cid in nz]
        wc = [wi for cid, wi in zip(act, w) if cid not in nz]
        if wn: wr["noisy"][r["round"]] = float(np.mean(wn))
        if wc: wr["clean"][r["round"]] = float(np.mean(wc))
    last = evals[-1]
    data[noise][method].append({
        "acc": last.get("acc"), "jain": last.get("jain"),
        "worst": last.get("client_acc_min"),
        "per_client": last.get("per_client_acc"), "noisy_ids": sorted(nz),
        "curve": curve, "wr": wr,
    })

NOISES = sorted(data.keys())
print(f"[figures] noise levels found: {NOISES}")
for n in NOISES:
    for m in METHODS:
        print(f"   n{n:02d} {m:9s}: {len(data[n][m])} seeds")


def stat(noise, method, key):
    xs = [d[key] for d in data[noise][method] if d.get(key) is not None]
    return (np.mean(xs), np.std(xs), len(xs)) if xs else (np.nan, np.nan, 0)


def curve_mean(noise, method, key):
    seeds = data[noise][method]
    if not seeds:
        return np.array([]), np.array([]), np.array([])
    allr = sorted(set().union(*[set(s["curve"][key]) for s in seeds]))
    mean = []; std = []
    for rd in allr:
        vs = [s["curve"][key][rd] for s in seeds if rd in s["curve"][key] and s["curve"][key][rd] is not None]
        mean.append(np.mean(vs)); std.append(np.std(vs))
    return np.array(allr), np.array(mean), np.array(std)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"   wrote {name}.png/.pdf")


PRETTY = {"acc": "Accuracy", "jain": "Jain fairness index", "worst": "Worst-client accuracy"}

# ---- 1. Grouped bars: metric at clean vs 40% noise ----
def grouped_bars(key, noises_to_show):
    noises_to_show = [n for n in noises_to_show if n in data]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(noises_to_show)); w = 0.26
    for i, m in enumerate(METHODS):
        means = [stat(n, m, key)[0] for n in noises_to_show]
        errs = [stat(n, m, key)[1] for n in noises_to_show]
        bars = ax.bar(x + (i - 1) * w, means, w, yerr=errs, capsize=3,
                      color=COLOR[m], label=LABEL[m], edgecolor="white", linewidth=0.8)
        for b, mv in zip(bars, means):
            if not np.isnan(mv):
                ax.text(b.get_x() + b.get_width()/2, mv + 0.012, f"{mv:.2f}",
                        ha="center", va="bottom", fontsize=8.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([f"{n}% noise" if n else "Clean" for n in noises_to_show])
    ax.set_ylabel(PRETTY[key]); ax.set_ylim(0, 1.05)
    ax.set_title(f"{PRETTY[key]} — OCTMNIST (3 seeds, mean±std)", fontsize=11)
    ax.legend(ncol=3, loc="upper right", fontsize=9.5)
    save(fig, f"bars_{key}")

for key in ["acc", "jain", "worst"]:
    grouped_bars(key, [0, 40])

# ---- 2. Learning curves (acc/jain/worst vs round) at a given noise ----
def learning_curves(key, noise):
    if noise not in data:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for m in METHODS:
        r, mu, sd = curve_mean(noise, m, key)
        if r.size == 0:
            continue
        ax.plot(r, mu, color=COLOR[m], lw=LW[m], ls=LSTYLE[m], marker=MARKER[m],
                markersize=4, markevery=3, label=LABEL[m])
        ax.fill_between(r, mu - sd, mu + sd, color=COLOR[m], alpha=0.12, linewidth=0)
        ax.text(r[-1] + 0.4, mu[-1], LABEL[m], color=COLOR[m], fontsize=9,
                va="center", fontweight="bold")
    ax.set_xlabel("Communication round"); ax.set_ylabel(PRETTY[key])
    cond = "clean" if noise == 0 else f"{noise}% label noise"
    ax.set_title(f"{PRETTY[key]} over training — {cond}", fontsize=11)
    ax.set_xlim(right=r[-1] + 5); ax.legend(fontsize=9.5, loc="lower right")
    save(fig, f"curve_{key}_n{noise:02d}")

for noise in [0, 40]:
    for key in ["acc", "jain", "worst"]:
        learning_curves(key, noise)

# ---- 3. Degradation curves: metric vs noise level ----
def degradation(key):
    if len(NOISES) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for m in METHODS:
        xs = [n for n in NOISES if stat(n, m, key)[2] > 0]
        mu = [stat(n, m, key)[0] for n in xs]
        sd = [stat(n, m, key)[1] for n in xs]
        ax.errorbar(xs, mu, yerr=sd, color=COLOR[m], lw=LW[m], ls=LSTYLE[m],
                    marker=MARKER[m], markersize=6, capsize=3, label=LABEL[m])
    ax.set_xlabel("Corrupted-client fraction (%)"); ax.set_ylabel(PRETTY[key])
    ax.set_title(f"{PRETTY[key]} vs. noise level — OCTMNIST", fontsize=11)
    ax.legend(fontsize=9.5)
    save(fig, f"degradation_{key}")

for key in ["acc", "jain", "worst"]:
    degradation(key)

# ---- 4. Combined 1x3 degradation panel ----
if len(NOISES) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, key in zip(axes, ["acc", "jain", "worst"]):
        for m in METHODS:
            xs = [n for n in NOISES if stat(n, m, key)[2] > 0]
            mu = [stat(n, m, key)[0] for n in xs]; sd = [stat(n, m, key)[1] for n in xs]
            ax.errorbar(xs, mu, yerr=sd, color=COLOR[m], lw=LW[m], ls=LSTYLE[m],
                        marker=MARKER[m], markersize=6, capsize=3, label=LABEL[m])
        ax.set_xlabel("Corrupted-client fraction (%)"); ax.set_title(PRETTY[key], fontsize=11)
    axes[0].legend(fontsize=9.5)
    fig.suptitle("Robustness to label noise — OCTMNIST (3 seeds)", fontsize=12, y=1.02)
    save(fig, "degradation_panel")

# ---- 5. Mechanism: aggregation weight to noisy vs clean clients over rounds (proposed @40%) ----
def mechanism(noise=40):
    seeds = data.get(noise, {}).get("proposed", [])
    if not seeds:
        return
    allr = sorted(set().union(*[set(s["wr"]["noisy"]) for s in seeds if s["wr"]["noisy"]]))
    if not allr:
        return
    def avg(kind):
        return [np.mean([s["wr"][kind][rd] for s in seeds if rd in s["wr"][kind]]) for rd in allr]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(allr, avg("clean"), color="#009E73", lw=2.4, marker="^", markevery=3,
            markersize=4, label="Clean clients")
    ax.plot(allr, avg("noisy"), color="#D55E00", lw=2.4, ls="--", marker="x", markevery=3,
            markersize=5, label="Corrupted clients")
    ax.set_xlabel("Communication round"); ax.set_ylabel("Mean aggregation weight")
    ax.set_title(f"Proposed down-weights corrupted clients ({noise}% noise)", fontsize=11)
    ax.legend(fontsize=9.5)
    save(fig, f"mechanism_weights_n{noise:02d}")

mechanism(40)

# ---- 6. Per-client accuracy distribution (strip) at 40% noise, last round ----
def per_client_strip(noise=40):
    if noise not in data:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    rng = np.random.default_rng(0)
    for i, m in enumerate(METHODS):
        seeds = data[noise][m]
        pts = []
        for s in seeds:
            if s["per_client"]:
                pts.extend(s["per_client"])
        if not pts:
            continue
        xj = i + (rng.random(len(pts)) - 0.5) * 0.28
        ax.scatter(xj, pts, s=26, color=COLOR[m], alpha=0.6, edgecolor="white", linewidth=0.5)
        ax.hlines(np.mean(pts), i - 0.28, i + 0.28, color="#222", lw=2)
        ax.text(i, min(pts) - 0.03, f"min={min(pts):.2f}", ha="center", fontsize=8, color="#555")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([LABEL[m] for m in METHODS])
    ax.set_ylabel("Per-client accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Per-client accuracy spread — {noise}% noise (black bar = mean)", fontsize=11)
    save(fig, f"per_client_strip_n{noise:02d}")

per_client_strip(40)

# ---- 7. Accuracy retention (clean -> 40% drop) ----
def retention():
    if 0 not in data or 40 not in data:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, m in enumerate(METHODS):
        c = stat(0, m, "acc")[0]; nz = stat(40, m, "acc")[0]
        drop = 100 * (c - nz) / c if c else np.nan
        b = ax.bar(i, drop, 0.6, color=COLOR[m])
        ax.text(i, drop + 0.4, f"{drop:.0f}%", ha="center", fontsize=9.5, color="#333")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([LABEL[m] for m in METHODS])
    ax.set_ylabel("Accuracy drop clean → 40% noise (%)")
    ax.set_title("Relative accuracy lost to 40% label noise (lower = more robust)", fontsize=10.5)
    save(fig, "accuracy_retention")

retention()

# ---- Tables: Markdown + LaTeX ----
def build_tables():
    md = ["# Paper tables — OCTMNIST, 3 seeds, last round (mean±std)\n"]
    tex = []
    for noise in NOISES:
        cond = "Clean (0% noise)" if noise == 0 else f"{noise}% noisy clients (60% label flip)"
        md.append(f"\n### {cond}\n")
        md.append("| Method | Accuracy | Jain index | Worst-client acc |")
        md.append("|---|---|---|---|")
        for m in METHODS:
            a = stat(noise, m, "acc"); j = stat(noise, m, "jain"); w = stat(noise, m, "worst")
            md.append(f"| {LABEL[m]} | {a[0]:.3f} ± {a[1]:.3f} | {j[0]:.3f} ± {j[1]:.3f} | {w[0]:.3f} ± {w[1]:.3f} |")
        tex.append(f"% {cond}")
        tex.append(r"\begin{tabular}{lccc}")
        tex.append(r"\toprule")
        tex.append(r"Method & Accuracy & Jain index & Worst-client acc \\")
        tex.append(r"\midrule")
        for m in METHODS:
            a = stat(noise, m, "acc"); j = stat(noise, m, "jain"); w = stat(noise, m, "worst")
            tex.append(f"{LABEL[m]} & ${a[0]:.3f}\\pm{a[1]:.3f}$ & ${j[0]:.3f}\\pm{j[1]:.3f}$ & ${w[0]:.3f}\\pm{w[1]:.3f}$ \\\\")
        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}"); tex.append("")
    open(os.path.join(RESDIR, "paper_tables.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(RESDIR, "paper_tables.tex"), "w").write("\n".join(tex) + "\n")
    print("   wrote results/paper_tables.md and .tex")

build_tables()
print(f"\n[figures] all outputs in {FIGDIR}")

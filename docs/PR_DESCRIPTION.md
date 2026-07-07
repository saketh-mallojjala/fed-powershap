# PR: Robust FL under unreliable clients — loss-based client down-weighting

_(Paste the body below into the GitHub PR. Base: `main` ← `feat/robust-shapfed-outperform`.)_

---

## Summary

Reworks the proposed federated-learning method for the **unreliable-client** regime and evaluates it honestly on a single, reproducible protocol (3 seeds, OCTMNIST).

**Proposed = FedDyn optimizer + loss-based client-quality down-weighting.** Each client is weighted by how well the clean global consensus fits its data (`w ∝ exp(−β·loss)`); corrupted clients (flipped labels) have high loss → low weight.

## Key results (3 seeds, last round)

| | Accuracy | Jain | Worst-client |
|---|---|---|---|
| Clean — FedDyn | 0.667 | 0.965 | 0.481 |
| Clean — Proposed | **0.695** | 0.958 | 0.452 |
| 40% noise — FedDyn | 0.530 ± 0.086 | 0.906 | 0.232 |
| 40% noise — Proposed | 0.518 ± 0.042 | 0.904 | 0.215 |

**Verdict (honest):** proposed **matches FedDyn** (slightly ahead clean, statistical tie under 40% noise) with **lower variance**, plus a **verified mechanism** that down-weights corrupted clients (weight ratio 0.69, 3 seeds). Parity + working robustness mechanism — not an across-the-board accuracy win.

## What changed
- **Loss-based detector** replaces ShapFed/CSSV cosine scoring, documented as a **negative result** (anti-discriminative under partial participation — up-weights corrupted clients).
- **Weight-consistent FedDyn** integration; provably identical to plain FedDyn under uniform weights.
- **Cosine LR schedule** (fixes late-round collapse) + **moderate weighting** (β=1.5, cap 0.30) to prevent weight collapse.
- Dropped Power-of-Choice selection.

## Deliverables
- `docs/FINAL_REPORT.md` — self-contained report.
- `docs/ROBUST_SHAPFED_RESULTS.md` — change log / Part II.
- `figures/paper/` — bars, learning curves, degradation panel, `mechanism_weights` (down-weighting evidence), `architecture` flow diagram, per-client spread.
- `results/paper_tables.{md,tex}` — paper tables.
- `scripts/run_noise_sweep.py`, `scripts/make_paper_figures.py`, `scripts/make_architecture_figure.py`.
- 18 result logs in `logs/`.

## Reproduce
```bash
python3 scripts/run_noise_sweep.py --noises 0.0 0.4 --seeds 0 1 2
python3 scripts/make_paper_figures.py
```

## Limitations
40%-noise accuracy is a tie (not a win); one dataset / 3 seeds (high variance). Next: noise sweep (20/60%), more seeds, ISIC/APTOS.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

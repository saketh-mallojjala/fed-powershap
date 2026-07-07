# FedPowerShap — Robust FL: Method, Mechanism & Multi-Seed Results (Part II)

_Continuation of `docs/RESULTS_ANALYSIS.md`. That document showed the proposed
method (Power-of-Choice + ShapFed-WA) could not beat FedDyn and recommended
dropping PoC, adding a robust contribution signal, and running multi-seed
noisy-client experiments. This document reports that work — including a signal
that **failed** and the one that **works** — with the final numbers. All results
are 3-seed, OCTMNIST, Dirichlet α=0.3, ResNet-18 @ 64px, one frozen protocol.
Fully reproducible via `scripts/run_noise_sweep.py`._

---

## 1. Recap — the starting point

From `RESULTS_ANALYSIS.md`: the proposed method lost to FedDyn (the strongest
baseline) — clean accuracy 0.507 vs 0.654, and under 40% label noise it was the
*worst* on fairness. Diagnosis: (a) contribution weighting can't beat size-
weighting on clean data (unbiased optimum); (b) the contribution signal was
noise-blind; (c) FedDyn and ShapFed were mis-integrated; (d) Power-of-Choice
selects the corrupted clients under noise; (e) constant LR caused late collapse.

---

## 2. What we changed (and what we learned)

### 2.1 Two engineering fixes (kept)
- **Dropped Power-of-Choice** → random selection (PoC selects the corrupted
  highest-loss clients under noise). `config.py` `proposed` preset.
- **Cosine LR schedule** applied uniformly to all methods → removes the constant-
  LR late-round collapse; metrics are read at the converged (last) round.
- **Weight-consistent FedDyn** (`_agg_feddyn`): the FedDyn drift-correction now
  tracks the same reweighted average that forms the model. Provably identical to
  legacy FedDyn under uniform weights (verified), so the baseline is unchanged.

### 2.2 The contribution signal — a negative result, then a fix
The core question was how to *detect* unreliable clients. We tried two signals.

**(a) ShapFed CSSV / robust median — FAILED.** Scoring each client by the cosine
of its classifier update against a (mean or coordinate-wise **median**)
consensus. **On OCTMNIST this signal is anti-discriminative**: corrupted clients
receive *higher* contribution scores and *more* aggregation weight (verified
across 3 seeds — noisy/clean weight ratio ≈ 1.1–1.3). Reason: under partial
participation (5 of 10 active, 40% corrupted), the active set often contains
2–3 corrupted clients, so the consensus direction is itself pulled toward the
corrupted updates. **This is a genuine finding**: cosine-of-update contribution
scoring does not identify label-noise clients here, regardless of the robust
reference. (Retained behind `--contrib-signal cssv` for the ablation.)

**(b) Loss-based detector — WORKS.** Weight each active client by the global
model's loss on its **own** data: `w ∝ exp(−β·(loss − min_loss))`, blended with
the size prior and capped. Corrupted clients (flipped labels) disagree with the
clean consensus → high loss → low weight. **Verified on OCTMNIST 40% noise (3
seeds): corrupted clients receive 0.156 mean weight vs 0.228 for clean (ratio
0.69)** — reliably down-weighted from round 1 on (see
`figures/paper/mechanism_weights_n40.*`). This is the mechanism the method now
uses (`config.py` `contrib_signal="loss"`, `server._loss_weights`).

### 2.3 Stability — weight-collapse fix
Aggressive loss-weighting (β=4, cap=0.5, blend=0.85) collapses aggregation onto
1–2 clients and destabilizes the global model (best-round 0.57 but last-round
crashes to 0.27–0.39 on some seeds). The final preset uses **moderate** settings
— **β=1.5, blend λ=0.5, weight cap 0.30** — which keep the noise suppression
while staying stable (worst seed's last-round 0.39 → 0.57, no collapse).

---

## 3. Protocol
OCTMNIST, α=0.3 non-IID; ResNet-18 @ 64px; 10 clients, 5 active/round, 2 local
epochs, SGD lr=1e-3 **cosine-decayed**, 40 rounds, 20k train cap; 40% of clients
have 60% of their training labels flipped (test labels clean); 3 seeds; last-round
metric. Methods: FedAvg, FedDyn (strongest baseline), Proposed.

---

## 4. Final results (3 seeds, mean±std, last round)

### Table 1 — Clean data (0% noise)
| Method | Accuracy | Jain | Worst-client |
|---|---|---|---|
| FedAvg | 0.645 ± 0.065 | 0.935 | 0.342 |
| FedDyn | 0.667 ± 0.031 | **0.965** | **0.481** |
| **Proposed** | **0.695 ± 0.030** | 0.958 | 0.452 |

### Table 2 — 40% noisy clients (60% label flip)
| Method | Accuracy | Jain | Worst-client |
|---|---|---|---|
| FedAvg | 0.448 ± 0.055 | 0.864 | 0.119 |
| FedDyn | **0.530 ± 0.086** | **0.906** | **0.232** |
| **Proposed** | 0.518 ± 0.042 | 0.904 | 0.215 |

Seed-matched at 40%: Proposed wins seed0 (0.467 vs 0.463) and seed1 (0.516 vs
0.475); FedDyn wins seed2 (0.651 vs 0.570, a seed where FedDyn is unusually
strong).

---

## 5. Honest verdict

- **Clean data:** Proposed ≥ FedDyn on accuracy (0.695 vs 0.667) with tighter
  variance; comparable fairness.
- **40% noise:** Proposed **ties** FedDyn on every metric (0.518 vs 0.530 accuracy
  — within one std; Jain 0.904 vs 0.906; worst-client 0.215 vs 0.232) and is
  **more stable** (±0.042 vs ±0.086), winning 2 of 3 seeds head-to-head.
- **Mechanism:** verified — Proposed reliably identifies and down-weights the
  corrupted clients (ratio 0.69); FedDyn/FedAvg have no such signal.

**This is a tie-with-the-strongest-baseline plus a working, interpretable
robustness mechanism — not a clean-accuracy knockout.** The defensible
contribution is the *combination*: matches/slightly-exceeds FedDyn, lower
variance, and an explicit client-quality signal (which also supports an
incentive-mechanism framing). It is honest and publishable as such; it is **not**
"beats all baselines on accuracy."

---

## 6. Caveats
- The 40%-noise accuracy is a **statistical tie**, not a win; one seed (FedDyn's
  strong seed2) still favors FedDyn.
- The clean-data edge (0.695 vs 0.667) is within seed variance — read it as parity.
- **One dataset (OCTMNIST), 3 seeds.** Generalization to ISIC/APTOS is pending.
- The loss detector assumes corrupted clients are a minority and that clean-
  consensus loss separates them; it degrades if corruption is the majority.

---

## 7. Next steps
- Noise sweep 20/60% for the degradation curve; extend to ISIC/APTOS (both cached).
- More seeds (5–10) to tighten the high-variance 40% estimate (one seed currently
  swings the mean).
- Mental-health tabular case study as an applied instance of the unreliable-client
  setting.

---

## 8. Reproduction & figures
```bash
python3 scripts/run_noise_sweep.py --noises 0.0 0.4 --seeds 0 1 2   # runs
python3 scripts/make_paper_figures.py                               # figures + tables
```
Figures: `figures/paper/` (bars, learning curves, degradation panel,
`mechanism_weights_n40` = the down-weighting evidence, per-client spread).
Tables: `results/paper_tables.md` / `.tex`.
Ablations: `--contrib-signal cssv` (the failed signal), `--loss-weight-beta`,
`--cssv-max-weight`, `--no-feddyn-weight-consistent`.

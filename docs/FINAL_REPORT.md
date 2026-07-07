# Robust, Fair Federated Learning under Unreliable Clients — Final Report

**Project:** FedPowerShap · **Dataset:** OCTMNIST (retinal OCT, 4-class), non-IID
Dirichlet α=0.3 · **Model:** ResNet-18 (ImageNet-pretrained) · **Seeds:** 3 ·
**Status:** self-contained results report, fully reproducible.

> **One-line summary.** The proposed method matches the strongest baseline
> (FedDyn) — slightly ahead on clean accuracy, a statistical tie under 40% label
> noise — with **lower variance** and, uniquely, a **verified mechanism that
> identifies and down-weights unreliable (corrupted) clients**. It is positioned
> honestly as *robustness + a working client-quality signal at parity accuracy*,
> not as an across-the-board accuracy win.

---

## 1. Problem and goal

Federated learning (FL) trains a shared model across many clients (e.g.
hospitals) without centralizing their data. A recurring real-world issue is
**unreliable clients** — some participants hold noisy or corrupted labels. The
standard aggregators (FedAvg, FedDyn) trust every client in proportion to its
data size, so corrupted clients degrade the global model.

The proposed method aims to **detect and down-weight unreliable clients** while
matching the accuracy of the strongest baseline on clean data. This report
establishes, on a single fair protocol with multiple seeds, (a) whether it works,
(b) by what mechanism, and (c) an honest assessment of where it wins, ties, and
loses.

---

## 2. Where the project started

An earlier evaluation (`docs/RESULTS_ANALYSIS.md`) found the original proposed
method (Power-of-Choice selection + ShapFed-WA aggregation) **losing** to FedDyn:
clean accuracy 0.507 vs 0.654, and worst-on-fairness under 40% noise. The
diagnosis identified five issues, all addressed here:

1. On clean data, contribution weighting **cannot** beat size-weighted averaging
   (it is the unbiased optimum) — so clean-accuracy superiority is not a winnable
   claim.
2. The contribution signal was **noise-blind**.
3. FedDyn and the contribution weighting were **mis-integrated**.
4. **Power-of-Choice** selects the highest-loss clients — which under label noise
   are exactly the corrupted ones.
5. A constant learning rate caused **late-round collapse**, and earlier
   comparisons used inconsistent, non-reproducible settings.

---

## 3. Methodology

**Protocol (frozen; identical for every method).** 10 clients, 5 active per
round, 2 local epochs, SGD lr=1e-3 with **cosine decay**, 40 rounds, training
pool capped at 20,000 (stratified), α=0.3 non-IID. Metrics read at the
**converged (last) round** (the LR schedule removes late collapse, so best≈last
and there is no cherry-picking).

**Unreliable-client model.** A fraction of clients (here 40%) have 60% of their
**training** labels randomly flipped to a different class; **test labels stay
clean**. The corrupted client set is seeded and identical across methods.

**Fairness metric.** Jain's index `J = (Σaᵢ)² / (n·Σaᵢ²)` over per-client
accuracies (from a held-out per-client test split), plus the **worst-client
accuracy** (the most under-served client) and its spread.

**Methods compared.** FedAvg (size-weighted averaging), **FedDyn** (dynamic
regularization — the strongest baseline in all prior tables), and the **Proposed**
method (below).

---

## 4. The proposed method

**Proposed = FedDyn optimizer + loss-based client down-weighting.**

1. **Local optimizer: FedDyn.** Each client trains with dynamic regularization
   that limits drift from the global model (the strongest baseline optimizer).
2. **Client-quality detector (the novel part): loss-based weighting.** At
   aggregation, each active client is scored by the **global model's loss on its
   own data**. Corrupted clients (flipped labels) disagree with the clean
   consensus and therefore have **high loss**; their aggregation weight is
   `w ∝ exp(−β·(loss − min_loss))`, blended with the size prior and capped so no
   single client dominates. Corrupted clients are thus **down-weighted**.
3. **Selection: random** (Power-of-Choice removed).
4. **Weight-consistent FedDyn.** The FedDyn drift-correction is made consistent
   with the reweighted average, so the two mechanisms cooperate rather than
   cancel. (Identical to standard FedDyn when weights are uniform — verified.)

**Configuration:** β=1.5, blend λ=0.5, weight cap 0.30 (see §6 for why moderate).

**Architecture (one communication round).** See
`figures/paper/architecture.png` — the standard steps (global broadcast → random
selection → local FedDyn training) feed the **novel robustness block**: the
loss-based client-quality detector → loss-based weights → weight-consistent
FedDyn aggregation → updated global model, repeated for 40 rounds.

```
      Global model wₜ  ──►  Random selection (5 of 10)  ──►  Local FedDyn training
                                                                     │  client updates wₖ
                                                                     ▼
   ┌──────────────────────── NOVEL: robustness mechanism ────────────────────────┐
   │  Client-quality detector:  lossₖ = L(wₜ ; client k's data)   (noisy ⇒ high)  │
   │                    │                                                          │
   │                    ▼                                                          │
   │  Loss-based weights: wₖ ∝ exp(−β·lossₖ), blend size prior, cap 0.30           │
   │                    │            (high loss ⇒ low weight ⇒ down-weighted)      │
   │                    ▼                                                          │
   │  Weight-consistent FedDyn aggregation  ──►  wₜ₊₁                              │
   └──────────────────────────────────────────────────────────────────────────────┘
                                     │  (cosine LR decay)
                                     ▼
                        Updated global model wₜ₊₁  ──►  repeat
```

---

## 5. A negative result: the Shapley/cosine signal does not work here

We first tried the natural ShapFed approach: score each client by the **cosine
similarity** of its classifier update to the consensus update (and a robust
**coordinate-wise median** variant). **On OCTMNIST this signal is
anti-discriminative** — corrupted clients receive *higher* contribution scores
and *more* weight (noisy/clean weight ratio ≈ 1.1–1.3, verified across 3 seeds).

**Why:** under partial participation (5 of 10 active, 40% corrupted), the active
set usually contains 2–3 corrupted clients, so the consensus direction is itself
pulled toward the corrupted updates — the robust-median assumption (corruption is
a minority *of the compared set*) breaks. This is a genuine, reportable finding:
**cosine-of-update contribution scoring cannot identify label-noise clients under
partial participation.** It motivated the switch to the loss detector.

*(Reproduce the failed signal with `--contrib-signal cssv`.)*

---

## 6. Mechanism validation (the loss detector works)

On the actual OCTMNIST 40%-noise runs (3 seeds), the proposed method assigns:

- **mean weight to corrupted clients = 0.156**
- **mean weight to clean clients = 0.228**
- **ratio = 0.69** — corrupted clients are reliably down-weighted, from round 1 on.

See `figures/paper/mechanism_weights_n40.*` (clean line held consistently above
the corrupted line throughout training).

**Stability note.** *Aggressive* down-weighting (β=4, cap=0.5) collapses
aggregation onto 1–2 clients and destabilizes the global model (last-round
accuracy crashes on some seeds). The **moderate** configuration (β=1.5, λ=0.5,
cap=0.30) preserves noise suppression while remaining stable (worst seed's
last-round 0.39 → 0.57, no collapse).

---

## 7. Results (3 seeds, mean ± std, last round)

### Table 1 — Clean data (0% noise)
| Method | Accuracy | Jain index | Worst-client acc |
|---|---|---|---|
| FedAvg | 0.645 ± 0.065 | 0.935 | 0.342 |
| FedDyn | 0.667 ± 0.031 | **0.965** | **0.481** |
| **Proposed** | **0.695 ± 0.030** | 0.958 | 0.452 |

### Table 2 — 40% noisy clients (60% label flip)
| Method | Accuracy | Jain index | Worst-client acc |
|---|---|---|---|
| FedAvg | 0.448 ± 0.055 | 0.864 | 0.119 |
| FedDyn | **0.530 ± 0.086** | **0.906** | **0.232** |
| **Proposed** | 0.518 ± 0.042 | 0.904 | 0.215 |

### Table 3 — 40% noise, seed-matched accuracy
| Seed | FedDyn | Proposed | Winner |
|---|---|---|---|
| 0 | 0.463 | 0.467 | Proposed |
| 1 | 0.475 | 0.516 | Proposed |
| 2 | 0.651 | 0.570 | FedDyn |
| **mean** | **0.530 ± 0.086** | **0.518 ± 0.042** | tie (Proposed lower variance) |

---

## 8. Verdict — is the method defensible?

**Yes, framed honestly as follows:**

- **Clean data:** Proposed ≥ FedDyn on accuracy (0.695 vs 0.667), tighter
  variance, comparable fairness.
- **40% label noise:** Proposed **ties** FedDyn on every metric (accuracy within
  one std; Jain 0.904 vs 0.906; worst-client 0.215 vs 0.232), is **more stable**
  (±0.042 vs ±0.086), and wins 2 of 3 seeds head-to-head.
- **Mechanism:** a **verified, interpretable client-quality signal** that
  down-weights corrupted clients (ratio 0.69) — which neither baseline has, and
  which additionally supports an **incentive-mechanism** framing (explicit
  contribution/quality scores per client).

**The defensible claim is:** *a robust, interpretable FL method that matches the
strongest baseline (FedDyn) on clean and noisy data, with lower variance and a
verified mechanism for identifying unreliable clients.* It is **not** "beats all
baselines on accuracy" — under 40% noise it is a statistical tie.

---

## 9. Limitations (stated up front)
- The 40%-noise accuracy is a **tie**, not a win; seed 2 (where FedDyn is unusually
  strong) still favors FedDyn.
- The clean-data edge (0.695 vs 0.667) is within seed variance — read it as parity.
- **Single dataset (OCTMNIST), 3 seeds** — variance is high; one seed can move the
  mean. More seeds and datasets are needed to firm up the claim.
- The detector assumes corrupted clients are a minority and that clean-consensus
  loss separates them; it will degrade if corruption becomes the majority.

---

## 10. Next steps
1. **Noise sweep (20% / 60%)** → degradation curve across corruption levels.
2. **More seeds (5–10)** to tighten the high-variance 40% estimate.
3. **Generalize** to ISIC and APTOS (both datasets already cached).
4. **Mental-health tabular case study** as an applied instance of the
   unreliable-client setting (incentive-mechanism framing).

---

## 11. How to reproduce and where to look
```bash
# 1. Run all methods on the frozen protocol (clean + 40% noise, 3 seeds)
python3 scripts/run_noise_sweep.py --noises 0.0 0.4 --seeds 0 1 2

# 2. Regenerate every figure + the paper tables
python3 scripts/make_paper_figures.py
```
**Outputs**
- Figures (PNG + PDF): `figures/paper/`
  - `bars_acc / bars_jain / bars_worst` — clean vs 40% grouped bars.
  - `curve_acc_n40 / curve_jain_n40 / curve_worst_n40` — training curves under noise.
  - `mechanism_weights_n40` — **the down-weighting evidence** (clean > corrupted).
  - `per_client_strip_n40` — per-client accuracy spread.
  - `degradation_*` — metric vs noise level (fills in as more noise levels run).
- Tables: `results/paper_tables.md` and `results/paper_tables.tex`.
- Method definition: `config.py` → `METHOD_PRESETS["proposed"]`.
- Ablation switches: `--contrib-signal cssv` (the failed signal),
  `--loss-weight-beta`, `--cssv-max-weight`, `--no-feddyn-weight-consistent`.

---

## 12. Companion documents
- `docs/RESULTS_ANALYSIS.md` — the original diagnosis (why the first method lost).
- `docs/ROBUST_SHAPFED_RESULTS.md` — the change log and Part II summary.
- This report (`docs/FINAL_REPORT.md`) — the self-contained final write-up.

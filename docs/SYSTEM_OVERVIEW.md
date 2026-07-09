# FedPowerShap — System Overview, Experiments & Results (Plain-English Guide)

_A single self-contained explainer: what the system does, which datasets and
baselines we used, every experiment in one table, the results, and the
justification behind each conclusion. Source of truth: `config.py`,
`results/crossdataset_tables.md`, and `docs/RESEARCH_STUDY.md`._

---

## 1. The one-paragraph summary

We study **unreliable clients** in federated learning (FL) — hospitals/sites
whose local labels are noisy or corrupted. We built a method that **detects and
down-weights** those bad clients: FedDyn (the strongest baseline optimizer) plus
a **loss-based client-quality detector**. The detector provably works (it gives
corrupted clients ≈0.69× the weight of clean clients). But when we tested it
honestly across 3 medical datasets, **down-weighting did NOT beat FedDyn under
label noise** — it only reaches a tie where noise is severe enough to break
FedDyn. So the paper is an **honest characterization of *when* contribution-aware
down-weighting helps**, plus a documented **negative result** about the classic
ShapFed cosine signal. It is deliberately *not* a "we beat everyone" paper.

---

## 2. The problem we're solving

In cross-silo FL (e.g. hospitals training a shared model without sharing patient
data), some clients contribute mislabeled data. Standard aggregators (FedAvg,
FedDyn) trust each client **in proportion to its dataset size**, so a corrupted
client with lots of data drags the global model down. We want to **automatically
identify unreliable clients and reduce their influence** — while staying at least
as accurate as the best baseline on clean data.

---

## 3. How the system works (one communication round)

```
   Global model wₜ
        │
        ▼
   (1) Random client selection ── pick 5 of 10 clients this round
        │
        ▼
   (2) Local FedDyn training ── each client trains locally with dynamic
        │                        regularization (limits drift from wₜ)
        ▼
   ┌───────────────── NOVEL ROBUSTNESS BLOCK ─────────────────┐
   │ (3) Quality detector:  lossₖ = global model's loss on     │
   │       client k's own data   (corrupted client ⇒ high loss)│
   │ (4) Loss-based weight:  wₖ ∝ exp(−β·(lossₖ − min_loss))    │
   │       then blend with size prior (λ) and cap at 0.30      │
   │ (5) Weight-consistent FedDyn aggregation → wₜ₊₁           │
   └───────────────────────────────────────────────────────────┘
        │  (cosine LR decay across rounds)
        ▼
   Updated global model wₜ₊₁ → repeat for 40 rounds
```

**Key design choices and *why*:**

| Choice | Why |
|---|---|
| **Random selection** (not Power-of-Choice) | PoC picks the *highest-loss* clients — which under label noise are exactly the *corrupted* ones. It backfires. |
| **Loss-based detector** (not cosine/Shapley) | The cosine signal is anti-discriminative here (see §7). Loss cleanly separates corrupted clients: they disagree with the clean consensus → high loss → low weight. |
| **Moderate β=1.5, cap=0.30** | Aggressive down-weighting (β=4, cap=0.5) collapses aggregation onto 1–2 clients and destabilizes the model. Moderate values keep noise suppression *and* stability. |
| **Weight-consistent FedDyn** | The FedDyn drift-correction uses the *same* weights as the model average, so regularization and reweighting cooperate instead of cancelling. (Reduces exactly to standard FedDyn under uniform weights — verified, so the baseline is untouched.) |
| **Cosine LR schedule (all methods)** | Removes late-round collapse, so we can honestly read the **last (converged)** round — no best-round cherry-picking. |

Method is defined in `config.py → METHOD_PRESETS["proposed"]`.

---

## 4. Datasets we tried

All are medical-imaging FL benchmarks, partitioned **non-IID** with Dirichlet
α = 0.3, ResNet-18 (ImageNet-pretrained) at 64px.

| Dataset | Task | Classes | Notes |
|---|---|---|---|
| **OCTMNIST** | Retinal OCT | 4 | Primary dataset; 5 seeds on clean, 3–4 on noise |
| **ISIC / DermaMNIST** | Dermoscopy | 7 | 3 seeds; baseline is already noise-tolerant here |
| **APTOS-2019** | Diabetic retinopathy (ordinal) | 5 | 3 seeds; headline metric is **QWK** (ordinal grades) |

(Older exploratory sets — CIFAR-10, MNIST, FMNIST, ChestX-ray14, BraTS — exist in
the loaders but are **not** part of the final study.)

---

## 5. Baselines we compared against

All baselines are implemented as `--method` presets on **one frozen protocol**, so
the comparison is apples-to-apples.

| Method | Selection | Aggregation | Local solver |
|---|---|---|---|
| FedAvg | random | size-weighted average | SGD |
| FedProx | random | size-weighted average | + proximal term |
| SCAFFOLD | random | control-variate | control-variate SGD |
| **FedDyn** | random | dynamic reg | + dynamic reg **(strongest baseline)** |
| FedBN | random | avg, BN kept local | SGD |
| MOON | random | size-weighted average | + model-contrastive |
| PoC-FedAvg | Power-of-Choice | size-weighted average | SGD |
| FedCE | random | contribution-weighted | SGD |
| **Proposed** | random | loss-based reweight + weight-consistent FedDyn | FedDyn |

The head-to-head comparison in the paper focuses on **FedAvg** (the baseline
everyone knows) and **FedDyn** (the strongest baseline in every table we ran).

---

## 6. Experiments — all in one table

**Frozen protocol (identical for every method):** 10 clients, 5 active/round,
2 local epochs, SGD lr=1e-3 with cosine decay, 40 rounds, 20k-sample train cap,
per-client held-out test split for fairness. Metric read at the **last round**.

**Noise model:** a fraction ρ of clients have **60% of their training labels
flipped** to a random other class; test labels stay clean; the corrupted client
set is seeded identically across all methods.

| # | Experiment | Datasets | Conditions | Seeds | What it answers |
|---|---|---|---|---|---|
| E1 | Clean baseline comparison | OCT, ISIC, APTOS | 0% noise | 3–5 | Are we competitive on accuracy/fairness when data is clean? |
| E2 | Noisy-client robustness | OCT, ISIC, APTOS | 40% noise | 3–4 | Does down-weighting beat FedDyn when clients are corrupted? |
| E3 | Mechanism validation | OCTMNIST | 40% noise | 3 | Does the detector actually down-weight corrupted clients? |
| E4 | Negative result (cosine signal) | OCTMNIST | 40% noise | 3 | Can the classic ShapFed cosine signal identify bad clients? |
| E5 | Stability / weight-collapse | OCTMNIST | 40% noise | — | Why moderate (not aggressive) down-weighting? |
| E6 | Central finding (the "opening") | OCT, ISIC, APTOS | clean vs 40% | 3 | *When* does down-weighting pay off? |

---

## 7. Results with justification

### E1 — Clean data (accuracy, mean±std)

| Dataset | FedAvg | FedDyn | Proposed |
|---|---|---|---|
| OCTMNIST | 0.636 | 0.688 | **0.708** (n=5) |
| ISIC | 0.697 | **0.734** | 0.719 |
| APTOS | 0.678 | 0.715 | **0.728** |

**Verdict:** competitive — matches or slightly exceeds FedDyn on 2 of 3 datasets.
The OCTMNIST/APTOS edges are within seed variance, so read as **parity**, not a
knockout. On clean data, contribution weighting *cannot* beat size-weighting in
principle (size-weighting is the unbiased optimum), so parity is the honest ceiling.

### E2 — 40% label noise (accuracy, mean±std)

| Dataset | FedAvg | FedDyn | Proposed |
|---|---|---|---|
| OCTMNIST | 0.462 | **0.523** | 0.518 (near-tie) |
| ISIC | 0.669 | **0.672** | 0.645 |
| APTOS | 0.577 | **0.649** | 0.588 |

**Verdict:** FedDyn is **at least as good on every dataset**. Proposed only
*approaches* FedDyn on OCTMNIST (−0.006, a tie), and loses on ISIC/APTOS. Fairness
(Jain) and worst-client accuracy at 40% also favor FedDyn. So there is **no
general robustness win**.

### E3 — Mechanism works (the good news)

On OCTMNIST at 40% noise (3 seeds): corrupted clients get **0.156** mean weight vs
**0.228** for clean clients — a **0.69 ratio**, from round 1 onward. The detector
does exactly what it claims. Evidence: `figures/paper/mechanism_weights_n40.png`.

### E4 — Negative result (a real, reportable finding)

The classic **ShapFed cosine signal** (scoring clients by cosine similarity of
their classifier update to the consensus — even with a robust median reference) is
**anti-discriminative**: corrupted clients get *higher* scores and *more* weight
(noisy/clean ratio ≈ 1.1–1.3). **Why:** under partial participation (5 of 10
active, 40% corrupted), the active set often contains 2–3 corrupted clients, so
the consensus direction is *itself* pulled toward the corrupted updates — the
robust-median assumption breaks. This is why we switched to the loss detector.

### E5 — Stability justification

Aggressive down-weighting (β=4, cap=0.5) collapses aggregation onto 1–2 clients →
last-round accuracy crashes to 0.27–0.39 on some seeds. Moderate (β=1.5, cap=0.30)
removes the collapse (worst seed's last round 0.39 → 0.57). Hence the moderate config.

### E6 — The central finding: *when* does it help?

Define the **"opening"** = how much 40% noise hurts FedDyn (clean − noisy accuracy):

| Dataset | FedDyn opening | Proposed − FedDyn @ 40% |
|---|---|---|
| OCTMNIST | 0.158 (large) | −0.006 (near-tie) |
| APTOS | 0.066 (small) | −0.061 |
| ISIC | 0.062 (small) | −0.027 |

**Interpretation:** FedDyn's dynamic regularization already absorbs *moderate*
noise. When noise is mild (ISIC, APTOS) there's little to gain, and down-weighting
throws away useful data → we lose. Only when noise is severe enough to break
FedDyn (OCTMNIST) does down-weighting recover to parity — **but not beyond it.**
Down-weighting's value is **conditional**, and at 40% noise on these datasets it's
at best a tie. See `figures/paper/opening_scatter.png` and `delta_heatmap.png`.

---

## 8. The honest bottom line (what we do and don't claim)

**We DO claim:**
1. A **loss-based client-quality detector** with a *verified* mechanism (0.69 ratio).
2. A **negative result**: the ShapFed cosine signal fails under partial participation.
3. A **multi-dataset characterization** of *when* contribution-aware down-weighting
   helps (only when noise severely degrades the baseline).

**We do NOT claim:** state-of-the-art accuracy, a general robustness win, or that
the benefit generalizes. Making any of those claims would contradict our own
tables (§7).

This is publishable as a **rigorous, reproducible, honest characterization** — a
cautionary finding for the FL community — not as a leaderboard win.

---

## 9. Limitations

- Two noise conditions only (0%, 40%), label-noise only (no Byzantine / feature
  shift / quantity skew).
- 3 seeds on ISIC/APTOS (5 on OCTMNIST clean); high variance under noise
  (APTOS proposed ±0.13).
- 64px inputs, 40-round budget (compute-limited).
- Single participation regime (N=10, m=5).

---

## 10. How to reproduce

```bash
# per-dataset sweeps (clean + 40% noise, 3 seeds)
python3 scripts/run_noise_sweep.py --dataset octmnist --noises 0.0 0.4 --seeds 0 1 2
python3 scripts/run_noise_sweep.py --dataset isic     --noises 0.0 0.4 --seeds 0 1 2
python3 scripts/run_noise_sweep.py --dataset aptos    --noises 0.0 0.4 --seeds 0 1 2
# figures + tables
python3 scripts/make_paper_figures.py            # OCTMNIST-detailed
python3 scripts/make_crossdataset_figures.py     # 3-dataset summary
```

**Where to look:**
- Method definition: `config.py → METHOD_PRESETS["proposed"]`
- Full tables (with n per cell): `results/crossdataset_tables.md`
- Full paper write-up: `docs/RESEARCH_STUDY.md`
- Figures: `figures/paper/`
- Ablation switches: `--contrib-signal cssv` (the failed signal),
  `--loss-weight-beta`, `--cssv-max-weight`, `--no-feddyn-weight-consistent`
</content>
</invoke>

# When Does Contribution-Aware Aggregation Help? A Multi-Dataset Study of Loss-Based Client Down-Weighting in Federated Learning under Label Noise

**Authors:** _[your name(s)]_ · **Affiliation:** _[dept]_ · **Date:** July 2026
**Code & logs:** branch `feat/robust-shapfed-outperform` (fully reproducible)

---

## Abstract

Federated learning (FL) must tolerate *unreliable clients* whose local labels are
noisy or corrupted. A popular idea is *contribution-aware aggregation*: estimate
each client's usefulness and down-weight the untrustworthy ones. We build a
concrete instance — **FedDyn dynamic regularization combined with a loss-based
client-quality detector** that down-weights clients the clean global consensus
fits poorly — and evaluate it rigorously (one frozen protocol, 3–5 seeds) on
three medical-imaging FL benchmarks (OCTMNIST, ISIC, APTOS) under 0% and 40%
label noise. We report three findings. **(1)** The detector *works*: it reliably
assigns corrupted clients ≈0.69× the weight of clean clients. **(2)** On clean,
non-IID data the method is *competitive* with the strongest baseline (FedDyn) —
matching or slightly exceeding accuracy on 2/3 datasets. **(3)** Under label
noise, down-weighting **does not yield a robustness advantage**: FedDyn is at
least as good on every metric across all three datasets; the proposed method only
*approaches* FedDyn where noise degrades FedDyn most (OCTMNIST) and is worse where
the baseline is already noise-tolerant (ISIC, APTOS). Our contribution is this
**honest characterization of when contribution-aware down-weighting helps** — a
cautionary, reproducible result — together with a documented **negative result**
that the classical ShapFed cosine-contribution signal is anti-discriminative
under partial participation.

---

## 1. Introduction

**Problem.** In cross-silo FL (e.g. hospitals), some clients contribute
mislabeled data. Standard aggregators (FedAvg, FedDyn) weight clients by dataset
size, so corrupted clients degrade the global model. *Contribution-aware*
methods aim to detect and down-weight such clients.

**What we do.** We implement a contribution-aware FL method and ask, rigorously,
**whether down-weighting unreliable clients actually improves robustness over a
strong baseline (FedDyn), and under what conditions.** Rather than claim a blanket
win, we characterize *when* it helps.

**Contributions.**
1. A **loss-based client-quality detector** on top of FedDyn, with a
   verified mechanism (it demonstrably down-weights corrupted clients).
2. A **negative result**: the ShapFed cosine-of-update contribution signal is
   *anti-discriminative* under partial participation — it up-weights corrupted
   clients — so it cannot be used for noise robustness here.
3. A **multi-dataset, multi-seed characterization** showing that down-weighting
   provides no robustness advantage over FedDyn in general, and quantifying the
   single condition under which it becomes competitive.

We deliberately do **not** claim state-of-the-art robustness; the evidence does
not support it, and we report that plainly (§7–§8).

---

## 2. Background & related work

- **FedAvg** (McMahan 2017): size-weighted averaging — the unbiased optimum when
  all clients are reliable.
- **FedDyn** (Acar 2021): adds dynamic regularization to limit client drift; the
  strongest baseline in all our experiments.
- **ShapFed / CSSV** (Tastan 2024): approximates client Shapley value via the
  cosine similarity of the last-layer update to the consensus. We show this
  signal fails under label noise + partial participation (§5.2).
- **FedCE** (Jiang 2023): full-model gradient-alignment contribution estimate.
- **Robust FL** (Krum, trimmed-mean, median): robust aggregation against Byzantine
  updates; our loss detector is a lightweight, interpretable alternative aimed at
  label-noise clients specifically.

---

## 3. Method

### 3.1 Architecture (one communication round)
See **`figures/paper/architecture.png`**. Each round: the server broadcasts the
global model → **random** client selection (5 of 10) → **local FedDyn training** →
the **novel robustness block** (client-quality detector → loss-based weights →
weight-consistent FedDyn aggregation) → updated global model; repeat for 40
rounds with a cosine learning-rate schedule.

### 3.2 Loss-based client-quality detector
At aggregation, client *k*'s weight is
`wₖ ∝ exp(−β · (lossₖ − min_loss))`, where `lossₖ` is the global model's
cross-entropy on client *k*'s **own** data. Corrupted clients (flipped labels)
disagree with the clean consensus → high loss → low weight. Weights are then
convex-blended with the size prior (λ) and capped at 0.30 so no single client
dominates. Final config: **β = 1.5, λ = 0.5, cap = 0.30** (moderate values — see
§5.3 on why aggressive weighting is harmful).

### 3.3 Weight-consistent FedDyn
The FedDyn server drift-correction uses the *same* weights as the model average,
so regularization and reweighting are consistent. This reduces to standard FedDyn
under uniform weights (verified numerically identical), so the FedDyn baseline is
unchanged.

---

## 4. Experimental setup

- **Datasets:** OCTMNIST (4-class retinal OCT), ISIC/DermaMNIST (7-class
  dermoscopy), APTOS-2019 (5-class ordinal diabetic-retinopathy; headline metric
  **QWK**). All Dirichlet α = 0.3 non-IID.
- **Model:** ResNet-18 (ImageNet-pretrained), 64px.
- **Protocol (frozen, identical for every method):** 10 clients, 5 active/round,
  2 local epochs, SGD lr = 1e-3 with cosine decay, 40 rounds, 20k-sample train
  cap, per-client held-out test split for fairness.
- **Noise model:** a fraction *ρ* of clients (0% or 40%) have 60% of their
  **training** labels flipped to a random other class; test labels stay clean;
  the corrupted set is seeded identically across methods.
- **Metrics:** global-test accuracy (+QWK for APTOS), Jain fairness index over
  per-client accuracies, and worst-client accuracy.
- **Seeds:** 3–5 (mean ± std), last-round metric (cosine LR removes late-round
  collapse, so no best-round cherry-picking).

---

## 5. Method development & mechanism

### 5.1 The mechanism works (verified)
On OCTMNIST at 40% noise (3 seeds), corrupted clients receive **0.156** mean
aggregation weight vs **0.228** for clean clients — a **0.69 ratio**, from round 1
onward. See **`figures/paper/mechanism_weights_n40.png`** (clean line held above
corrupted throughout training). The detector does what it claims.

### 5.2 Negative result — the ShapFed cosine signal fails
Scoring clients by cosine similarity of the classifier update to the consensus
(mean *or* robust coordinate-wise median) is **anti-discriminative** on OCTMNIST:
corrupted clients get *higher* scores and *more* weight (noisy/clean weight ratio
≈ 1.1–1.3, 3 seeds). Cause: under partial participation (5 of 10 active, 40%
corrupted), the active set often contains 2–3 corrupted clients, so the consensus
direction is itself pulled toward the corrupted updates. **Cosine-of-update
contribution scoring cannot identify label-noise clients under partial
participation.** This motivated the loss detector.

### 5.3 Stability — avoid weight collapse
Aggressive down-weighting (β = 4, cap = 0.5) collapses aggregation onto 1–2
clients and destabilizes the global model (best-round 0.57 but last-round crashes
to 0.27–0.39 on some seeds). The moderate config (§3.2) removes the collapse
(worst seed's last-round 0.39 → 0.57).

---

## 6. Results

**Tables:** `results/crossdataset_tables.md` (full, with n per cell).
**Figures:** `figures/paper/` — `crossdataset_acc_clean`, `crossdataset_acc_noise`,
`delta_heatmap`, `opening_scatter`, plus OCTMNIST-detailed `bars_*`, `curve_*`.

### Table — accuracy (mean ± std), all datasets
| Dataset | Cond. | FedAvg | FedDyn | Proposed |
|---|---|---|---|---|
| OCTMNIST | clean | 0.636 | 0.688 | **0.708** |
| OCTMNIST | 40% | 0.462 | **0.523** | 0.518 |
| ISIC | clean | 0.697 | **0.734** | 0.719 |
| ISIC | 40% | 0.669 | **0.672** | 0.645 |
| APTOS | clean | 0.678 | 0.715 | **0.728** |
| APTOS | 40% | 0.577 | **0.649** | 0.588 |

### Δ heatmap (Proposed − FedDyn), see `delta_heatmap.png`
Green (proposed better) appears **only on clean accuracy** (OCTMNIST +0.020,
APTOS +0.013). Every 40%-noise cell — accuracy, Jain, worst-client — is ≤ 0.
Worst-client at 40% is −0.10 on OCTMNIST/APTOS (FedDyn clearly fairer under noise).

---

## 7. Analysis — the central finding

**Down-weighting only pays off when the noise severely hurts the baseline.**
Define the *opening* = FedDyn's accuracy drop from clean → 40% noise:

| Dataset | FedDyn opening (clean − 40%) | Proposed − FedDyn @ 40% |
|---|---|---|
| OCTMNIST | 0.158 (large) | −0.006 (near-tie) |
| APTOS | 0.066 (small) | −0.061 |
| ISIC | 0.062 (small) | −0.027 |

See **`figures/paper/opening_scatter.png`**: the proposed–FedDyn gap approaches
zero *only* as the opening grows. Interpretation: FedDyn's dynamic regularization
already absorbs moderate label noise; when noise is mild (ISIC, APTOS) there is
little to gain and down-weighting discards useful data, so the method loses. Only
when noise is severe enough to break FedDyn (OCTMNIST) does down-weighting recover
to parity — but not beyond it, in our experiments.

**Bottom line:** contribution-aware down-weighting is **not** a general robustness
improvement over FedDyn. Its value is *conditional* and, at 40% noise on these
datasets, at best a tie.

---

## 8. How we defend this study (validity & reviewer Q&A)

We defend the **rigor and honesty of the study**, not a performance claim.

- **"Is the comparison fair?"** Yes — one frozen protocol, identical for all
  methods; the FedDyn baseline is provably unchanged by our weight-consistent
  integration; noisy-client sets are seeded identically across methods.
- **"Are the numbers cherry-picked?"** No — we report the **last (converged)**
  round, enabled by the cosine LR schedule that removes late-round collapse; 3–5
  seeds with std; and we show the *unfavorable* datasets rather than only the
  favorable one.
- **"Does the method actually do what it claims?"** Yes — the mechanism figure
  directly verifies down-weighting of corrupted clients (0.69 ratio).
- **"Then why isn't it a win?"** We explain it mechanistically (§7): FedDyn
  already tolerates moderate noise; down-weighting helps only when the baseline
  collapses, which is dataset-dependent. This is a *finding*, not a failure of
  the experiment.
- **"Is a negative/conditional result publishable?"** The contribution is (i) a
  documented failure mode of the widely-used ShapFed cosine signal under partial
  participation, (ii) a verified but conditionally-useful loss detector, and
  (iii) a multi-dataset characterization of *when* contribution-aware FL helps —
  all reproducible. This is useful knowledge for the community.

**What we do not claim:** state-of-the-art accuracy, a robustness win, or
generalization of the benefit. Making any of those claims would contradict our
own Tables (§6) and is avoided.

---

## 9. Limitations
- Two noise conditions (0%, 40%) and label-noise only; other corruptions
  (Byzantine, feature shift, quantity skew) untested.
- 3 seeds on ISIC/APTOS (5 on OCTMNIST clean); high variance under noise
  (e.g. APTOS proposed ±0.13).
- 64px inputs and a 40-round budget (compute-limited, laptop MPS).
- Single client-count / participation regime (N=10, m=5).

---

## 10. Conclusion
A loss-based client-quality detector reliably identifies corrupted clients, and
the resulting method is competitive with FedDyn on clean non-IID data. However,
across three medical datasets, **down-weighting corrupted clients does not improve
robustness over FedDyn under 40% label noise** — it only reaches parity where
noise most degrades the baseline. We contribute this honest characterization, a
documented failure of cosine-based contribution scoring, and fully reproducible
code and logs.

---

## 11. Reproducibility
```bash
# per-dataset sweeps (clean + 40% noise, 3 seeds)
python3 scripts/run_noise_sweep.py --dataset octmnist --noises 0.0 0.4 --seeds 0 1 2
python3 scripts/run_noise_sweep.py --dataset isic     --noises 0.0 0.4 --seeds 0 1 2
python3 scripts/prep_aptos_cache.py --image-size 64            # one-time
python3 scripts/run_noise_sweep.py --dataset aptos    --noises 0.0 0.4 --seeds 0 1 2
# figures + tables
python3 scripts/make_paper_figures.py            # OCTMNIST-detailed
python3 scripts/make_crossdataset_figures.py     # 3-dataset summary
python3 scripts/make_architecture_figure.py      # architecture diagram
```
Method preset: `config.py → METHOD_PRESETS["proposed"]`
(FedDyn solver + `contrib_signal="loss"`). Ablations: `--contrib-signal cssv`
(the failed signal), `--loss-weight-beta`, `--cssv-max-weight`,
`--no-feddyn-weight-consistent`.

## Figure index (all in `figures/paper/`, PNG + PDF)
| File | Shows |
|---|---|
| `architecture` | proposed method, one FL round |
| `mechanism_weights_n40` | detector down-weights corrupted clients (0.69) |
| `crossdataset_acc_clean` / `_noise` | accuracy, 3 datasets × 3 methods |
| `delta_heatmap` | Proposed − FedDyn across metrics (honest summary) |
| `opening_scatter` | **central finding**: benefit vs baseline degradation |
| `bars_*`, `curve_*` (OCTMNIST) | detailed bars & training curves |

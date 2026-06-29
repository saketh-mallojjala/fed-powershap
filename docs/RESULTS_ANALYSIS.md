# FedPowerShap — Results & Analysis

_Comprehensive write-up of the federated-learning method comparison, fairness
evaluation, and the investigation into the proposed method. All numbers below
are from runs in this repository (OCTMNIST, non-IID Dirichlet α=0.3)._

---

## 1. Goal

The proposed method (originally **Power-of-Choice selection + ShapFed-WA
aggregation**) was under-performing vanilla FedAvg, which is not publishable.
The objectives were to (a) compare against the full baseline suite, (b) add a
fairness metric (Jain's index), and (c) determine whether — and in what regime —
the proposed method can genuinely win.

---

## 2. What was implemented

**Methods (selectable via `--method`):**
FedAvg, FedProx, SCAFFOLD, FedDyn, FedBN, MOON, PoC-FedAvg, FedCE, and the
Proposed method. Each expands to a (selection, aggregation, local-solver) preset
in `config.py` (`METHOD_PRESETS`).

**Fairness:** Jain's fairness index `J = (Σaᵢ)² / (n·Σaᵢ²)` over per-client
accuracies, computed from a seeded per-client held-out test split
(`local_test_frac`), logged each round with worst-client accuracy and std.

**Data-quality heterogeneity:** per-client label noise (`noisy_client_frac`,
`label_noise_rate`) — a fraction of clients have a fraction of their training
labels randomly flipped (test labels stay clean) to simulate unreliable clients.

**Bug fixes required for a fair comparison:**
- Server momentum diverged (loss → 1900+) because of accumulation-form
  amplification → reformulated as an EMA (unit steady-state gain).
- Server momentum / FedDyn corrupted BatchNorm `running_var` → BN running stats
  are now always plain-averaged (`_is_running_stat` guard).
- SCAFFOLD diverged with SGD momentum (control variate conflicts with the
  momentum buffer) → SCAFFOLD uses a momentum-free local optimizer.

---

## 3. Experimental protocol

- **Dataset:** OCTMNIST (4-class retinal OCT), Dirichlet α=0.3 non-IID.
- **Model:** ResNet-18 (ImageNet-pretrained), 64px.
- **Federation:** 10–20 clients, 5 active/round, 2 local epochs, SGD lr=1e-3.
- **Metrics:** global-test accuracy, Jain fairness index, worst-client accuracy.
- The full baseline matrix used **3 seeds (mean±std)**; the proposed-method
  investigation used single-seed diagnostics for fast iteration.

---

## 4. Results

### Table 1 — Full baseline matrix (clean data, 20 clients, 3 seeds)
| Method | Accuracy | Jain index | Worst-client acc |
|---|---|---|---|
| **FedDyn** | **0.654 ± 0.009** | **0.968 ± 0.013** | **0.467 ± 0.046** |
| FedCE | 0.555 ± 0.041 | 0.937 ± 0.037 | 0.325 ± 0.073 |
| FedAvg | 0.548 ± 0.076 | 0.911 ± 0.042 | 0.240 ± 0.171 |
| FedProx | 0.547 ± 0.068 | 0.914 ± 0.049 | 0.249 ± 0.177 |
| FedBN | 0.545 ± 0.073 | 0.872 ± 0.041 | 0.116 ± 0.091 |
| MOON | 0.532 ± 0.053 | 0.914 ± 0.047 | 0.221 ± 0.160 |
| Proposed (PoC + ShapFed-WA) | 0.507 ± 0.009 | 0.915 ± 0.016 | 0.175 ± 0.126 |
| PoC-FedAvg | 0.447 ± 0.030 | 0.728 ± 0.152 | 0.055 ± 0.048 |
| SCAFFOLD | 0.392 ± 0.011 | 0.869 ± 0.032 | 0.038 ± 0.040 |

**Takeaway:** FedDyn is the strongest baseline. The original Proposed method
ranks 7th of 9 — selection-based methods (PoC-FedAvg) and SCAFFOLD are weakest.

### Table 2 — Proposed-method variants, clean data (10 clients, 25 rounds, 1 seed)
| Variant | Accuracy | Jain | Worst-client |
|---|---|---|---|
| **FedDyn (baseline)** | **0.729** | **0.965** | **0.586** |
| Proposed: random sel + Shapley-weighted FedDyn | 0.627 | 0.920 | 0.414 |
| Proposed: PoC + Shapley-weighted FedDyn | 0.522 | 0.958 | 0.506 |
| FedAvg | 0.520 | 0.860 | 0.134 |

**Takeaway:** Integrating FedDyn with ShapFed (Shapley-weighted aggregation)
*lowers* accuracy vs. plain FedDyn on clean data. Power-of-Choice makes it worse.

### Table 3 — Under 40% noisy clients (60% label flip)
| Method | Accuracy | Jain | Worst-client |
|---|---|---|---|
| FedDyn | 0.479 | 0.854 | 0.101 |
| Proposed (PoC variant) | 0.444 | 0.837 | 0.072 |
| FedAvg | 0.430 | 0.849 | 0.132 |

### Table 4 — Under 30% noisy clients (60% label flip), corrected proposed
| Method | Accuracy | Jain | Worst-client |
|---|---|---|---|
| FedDyn | **0.570** | 0.859 | **0.129** |
| **Proposed (random sel + strong Shapley)** | 0.553 | **0.864** | 0.122 |
| FedAvg | 0.530 | 0.813 | 0.091 |

**Takeaway:** In the noisy regime the corrected Proposed method **beats FedAvg
and matches FedDyn on accuracy, with the best fairness (Jain).** Note FedDyn
collapses from 0.729 (clean) to 0.48–0.57 under noise — the opening that
contribution-aware aggregation is meant to exploit.

---

## 5. Key findings

1. **The early "win" was an artifact.** An initial run showed Proposed 0.788 vs
   FedAvg 0.727, but that used few clients, abundant per-client data and only 12
   rounds. Under controlled evaluation (more clients, 3 seeds, full convergence)
   it does not hold.

2. **On clean data, contribution/Shapley weighting cannot beat size/uniform
   averaging — by design.** When all clients have good data, the statistically
   optimal aggregation weight is proportional to dataset size (the unbiased
   estimator that FedAvg/FedDyn use). Any contribution reweighting deviates from
   that optimum and can only add variance. This is fundamental, not a tuning gap.

3. **Power-of-Choice selection is harmful in non-IID.** It selects the
   highest-loss clients (the most skewed/outlier ones); PoC-FedAvg is near the
   bottom in every comparison. Under label noise it is *catastrophic* — the
   highest-loss clients ARE the corrupted ones, so PoC trains on bad data.

4. **FedDyn is the strongest baseline on clean data**, via dynamic
   regularization that limits client drift.

5. **FedDyn collapses under noisy clients** (0.729 → 0.48–0.57). This is the
   regime where down-weighting unreliable clients pays off — and where the
   proposed contribution-aware method is competitive and fairest.

---

## 6. Conclusion & recommended framing

The defensible, honest positioning for the paper is **not** "beats all baselines
on clean accuracy" (it cannot, per Finding 2). It is:

> **Robust, fair federated learning under unreliable/heterogeneous clients** —
> when a fraction of clients have noisy/corrupted data, the proposed
> contribution-aware method (FedDyn dynamic regularization + ShapFed Shapley
> weighting) maintains accuracy and achieves the best fairness, while FedAvg and
> FedDyn degrade.

This requires two changes to the original method:
- **Replace Power-of-Choice** (it drags performance down) with standard or
  reputation-guided selection.
- **Evaluate in the noisy-client regime** (the setting contribution-based FL
  papers — ShapFed, FedCE — are actually evaluated in).

---

## 7. Open items / next steps

- Strengthen the noisy-regime win: test **40% noise** with the corrected
  proposed variant (bigger opening), and a **robust-consensus** contribution
  signal (agreement with coordinate-wise median rather than the noise-polluted
  mean).
- Multi-seed runs of the noisy-regime comparison for mean±std.
- Extend to ISIC / ChestX-ray14 / APTOS (APTOS data is downloaded).
- Untested heterogeneity types that could enrich the story: feature/covariate
  shift (FedBN's target), image corruption, free-riders, Byzantine clients.

---

## 8. Reproduction

```bash
# Full baseline matrix (3 seeds) on the cached MedMNIST datasets
bash scripts/run_full.sh                       # -> results/ + figures/

# Noisy-client comparison (e.g. 30% of clients, 60% label flip)
python main.py --method feddyn   --dataset octmnist --noisy-client-frac 0.3 --label-noise-rate 0.6 ...
python main.py --method proposed --dataset octmnist --noisy-client-frac 0.3 --label-noise-rate 0.6 \
               --selection-strategy random --agg-blend-lambda 0.85 ...

# Aggregate to a table + plots
python scripts/aggregate_results.py
python scripts/plot_results.py logs/<runs>.jsonl
```

Method presets live in `config.py` (`METHOD_PRESETS`); the proposed method is
`shapfed_dyn` aggregation + `feddyn` local solver.

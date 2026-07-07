# FedPowerShap — Robust ShapFed: Method Changes & Multi-Seed Results (Part II)

_Continuation of `docs/RESULTS_ANALYSIS.md`. That document concluded the proposed
method could not win on clean accuracy and recommended (§6–§7): drop Power-of-
Choice, add a **robust-consensus** contribution signal (coordinate-wise median
instead of the noise-polluted mean), and run a **multi-seed 40%-noise**
comparison. This document reports exactly that work — the implementation, the
new results, and the before/after improvement. All numbers are from runs in this
repository (OCTMNIST, non-IID Dirichlet α=0.3, ResNet-18 @ 64px)._

---

## 1. Recap — where we were

From `RESULTS_ANALYSIS.md`, on a like-for-like comparison the proposed method
(originally Power-of-Choice selection + ShapFed-WA aggregation) was **losing**:

- **Clean data (3 seeds, Table 1):** Proposed 0.507 accuracy — 7th of 9 methods,
  behind FedDyn (0.654) and even FedAvg (0.548).
- **40% label noise (1 seed, Table 3):** Proposed 0.444 / Jain 0.837 /
  worst-client 0.072 — the **worst fairness** of the three, losing to FedDyn
  (0.479) on every axis.

The diagnosis (Findings §5): (a) on clean data, contribution weighting cannot
beat size-weighted averaging — it is the unbiased optimum, so this was an
unwinnable game; (b) the contribution signal was **noise-blind** (scored each
client against the *mean* update, which is polluted by the corrupted clients);
(c) FedDyn and ShapFed were **mis-integrated** (FedDyn's correction assumed
uniform averaging while aggregation reweighted); (d) **Power-of-Choice** selects
the highest-loss clients, which under noise *are* the corrupted ones; and (e) a
constant LR caused a late-round collapse.

---

## 2. What we changed

Five changes, four to the method + one to the evaluation. Recover the old
behaviour for ablation via CLI flags (noted per item).

### 2.1 Robust-consensus contribution signal — `federated/shapley.py`
CSSV now scores each client against a **robust consensus** direction (coordinate-
wise **median** or trimmed mean) instead of the mean of client updates
(`_consensus_delta`, `compute_cssv(reference=...)`). Under noise the mean is
dragged toward the corrupted clients, so they look "aligned" and keep weight; the
median is unaffected by a minority of corrupted clients, so they score low and
get down-weighted. Config: `cssv_reference` (`mean`|`median`|`trimmed`),
`cssv_trim_frac`. _Ablate: `--cssv-reference mean`._

### 2.2 Principled FedDyn ⊕ ShapFed integration — `federated/server.py`
FedDyn's drift-correction accumulator `h` now tracks the **same** Shapley-
weighted average that forms the model (`_agg_feddyn`, weight-consistent update),
instead of a uniform client sum. Previously the model term was reweighted but the
correction was not, so the two mechanisms partly cancelled. Proven identical to
legacy FedDyn under uniform weights (plain-FedDyn accuracy unchanged, verified),
so only `shapfed_dyn` is affected. Config: `feddyn_weight_consistent`.
_Ablate: `--no-feddyn-weight-consistent`._

### 2.3 Dropped Power-of-Choice — `config.py`
The `proposed` preset now uses **random** selection (`poc_anneal=0`,
`reputation_weight=0`) with a strong Shapley blend (`agg_blend_lambda=0.85`) and
a weight cap (`cssv_max_weight=0.5`). _Ablate the old selection:
`--selection-strategy pow_d --poc-anneal 0.5 --reputation-weight 0.5`._

### 2.4 LR schedule (stability) — `federated/server.py`, `federated/client.py`
Optional cosine local-LR decay (`_round_lr`, threaded into every solver) curbs
the constant-LR late-round collapse that afflicted the earlier runs. Applied
uniformly to **all** methods so the comparison stays fair. Config:
`lr_schedule=cosine`, `lr_min_frac`.

### 2.5 One frozen, reproducible protocol (evaluation)
All methods run on an identical protocol via `scripts/run_noise_sweep.py`
(§5) with **3 seeds**, and metrics are read at the **converged/last round** (the
LR fix makes best≈last, so no cherry-picking). The earlier comparisons mixed
client counts / round counts / configs and were not reproducible.

### Mechanism check (does the median actually help?)
Direct test on heavily-noisy data — average aggregation weight given to noisy vs
clean clients:

| CSSV reference | weight[noisy] / weight[clean] |
|---|---|
| mean (original) | **1.08** — noisy clients get *more* weight |
| median (new) | **0.93** — noisy clients get *less* weight |

The mean *rewards* corrupted clients; the median suppresses them. This is the
lever the noisy-regime result depends on, and it fires as designed.

---

## 3. Experimental protocol (frozen)

- **Dataset:** OCTMNIST (4-class), Dirichlet α=0.3 non-IID.
- **Model:** ResNet-18 (ImageNet-pretrained), 64px.
- **Federation:** 10 clients, 5 active/round, 2 local epochs, SGD lr=1e-3 with
  **cosine decay**, 40 rounds, train pool capped at 20k (stratified).
- **Noise:** 40% of clients have 60% of their training labels flipped (test
  labels stay clean); identical seeded assignment across methods.
- **Seeds:** 0, 1, 2 (mean±std). **Metric:** last-round.
- **Methods compared:** FedAvg, FedDyn (strongest baseline), Proposed.

---

## 4. New results (OCTMNIST, 3 seeds, mean±std, last round)

### Table 1 — Clean data (0% noise)
| Method | Accuracy | Jain | Worst-client |
|---|---|---|---|
| FedAvg | 0.645 ± 0.065 | 0.935 | 0.342 |
| **FedDyn** | **0.667 ± 0.031** | **0.965** | **0.481** |
| Proposed | **0.667 ± 0.055** | 0.964 | 0.452 |

### Table 2 — 40% noisy clients (60% label flip)
| Method | Accuracy | Jain | Worst-client |
|---|---|---|---|
| FedAvg | 0.448 ± 0.055 | 0.864 | 0.119 |
| FedDyn | **0.530 ± 0.086** | 0.906 | 0.232 |
| **Proposed** | 0.526 ± 0.131 | **0.921** | **0.284** |

**Takeaways:**
- **Accuracy parity with FedDyn.** Proposed matches the strongest baseline on
  clean data (0.667 = 0.667) and ties it under 40% noise (0.526 vs 0.530, well
  within the seed std). It clearly beats FedAvg under noise.
- **Fairness win under noise.** Proposed has the **best Jain index (0.921)** and
  the **best worst-client accuracy (0.284)** — +22% over FedDyn (0.232) and +139%
  over FedAvg (0.119). It lifts the worst-off client the most while matching the
  best overall accuracy.

---

## 5. Before → after (the improvement)

| Setting | Metric | Before (RESULTS_ANALYSIS) | After (this work) |
|---|---|---|---|
| Clean | Proposed acc vs FedDyn | 0.507 vs 0.654 — **loses by ~15 pts** | 0.667 vs 0.667 — **parity** |
| 40% noise | Proposed acc vs FedDyn | 0.444 vs 0.479 — loses | 0.526 vs 0.530 — **tie** |
| 40% noise | Proposed worst-client | 0.072 — worst of three | **0.284 — best of three** |
| 40% noise | Proposed Jain | 0.837 — worst of three | **0.921 — best of three** |

The proposed method moved from **clearly losing to FedDyn on both accuracy and
fairness** to **matching it on accuracy and beating it on fairness** — i.e. it now
improves fairness/robustness under unreliable clients **without sacrificing
accuracy**. This is the defensible positioning `RESULTS_ANALYSIS.md` §6 argued
for, now backed by multi-seed evidence.

---

## 6. Honest caveats

- **Accuracy is parity, not a win.** Under noise, Proposed (0.526) does not
  exceed FedDyn (0.530); the claim is *fairness gain at equal accuracy*, not
  accuracy superiority. Say so.
- **High variance.** Proposed's 40%-noise accuracy std is large (±0.131) — larger
  than FedDyn's (±0.086). Reducing this is an open item.
- **Protocol contributes to the absolute lift.** Part of the clean-data jump
  (0.507→0.667) reflects the fairer protocol + LR schedule, which lifted all
  methods. The robust signal itself is what changes the *standing relative to
  FedDyn* and the fairness metrics.
- **One dataset, 3 seeds.** OCTMNIST only, so far.
- **Robustness has a boundary.** The median consensus is robust only while
  corrupted clients are a *minority*. 40% is inside that range; at ≥50% corruption
  the median itself is poisoned and the method should degrade — a clean, honest
  limit worth reporting rather than hiding.

---

## 7. Next steps

- **Noise sweep 20/30/60%** for the degradation curve (expected: FedDyn's line
  crashes as noise rises while Proposed's fairness holds; the 60% point should
  show the honest breakdown of the median).
- **Ablation** attributing the fairness gain to each change (median vs weight-
  consistent FedDyn vs dropped-PoC), via the CLI flags in §2.
- **Reduce the 40%-noise accuracy variance.**
- **Generalize** to ISIC and APTOS (both cached), 3 seeds.
- **Mental-health case study** (tabular + MLP) as an applied validation of the
  unreliable-client setting.

---

## 8. Reproduction

```bash
# Full 3-seed sweep on the frozen protocol (clean + 40% noise)
python3 scripts/run_noise_sweep.py --noises 0.0 0.4 --seeds 0 1 2

# A single run (e.g. proposed at 40% noise)
python3 main.py --method proposed --dataset octmnist --seed 0 \
  --noisy-client-frac 0.4 --label-noise-rate 0.6 --lr-schedule cosine \
  --model resnet18 --image-size 64 --num-clients 10 --num-rounds 40 \
  --max-train-samples 20000 --local-test-frac 0.2 --eval-every 2

# Ablations (recover the old proposed behaviour piece by piece)
#   old contribution signal:   --cssv-reference mean
#   old FedDyn integration:     --no-feddyn-weight-consistent
#   old selection:              --selection-strategy pow_d --poc-anneal 0.5 --reputation-weight 0.5
```

The proposed method is defined by the `proposed` preset in `config.py`
(`shapfed_dyn` aggregation + `feddyn` solver + median consensus + weight-
consistent FedDyn + random selection).

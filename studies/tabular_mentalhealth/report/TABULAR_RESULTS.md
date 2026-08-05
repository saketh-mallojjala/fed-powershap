# Tabular mental-health case study — first results

_Kaggle Student Depression Dataset, 10 clients, 10 seeds, one frozen protocol.
Run in response to the review instruction "try with a small Kaggle mental
dataset". This reports the pre-registered tests from
`docs/INCENTIVE_PROPOSAL.md` §10, including the one that failed._

Reproduce: `scripts/tabular_ceiling.py`, `scripts/run_tabular_sweep.py`,
`scripts/aggregate_tabular.py` (see `../README.md`). Full tables:
`../tables/tabular_tables.md`; centralized ceiling: `../tables/tabular_ceiling.json`;
raw run logs: `../logs/`.

---

## 1. Bottom line

| # | Pre-registered test | Result |
|---|---|---|
| **Ranking** | Does the loss detector correctly separate corrupted from clean clients on tabular data? (C3: detection AUC ≥ 0.85) | **PASSES** — AUC 0.917 @20% noise, 0.874 @40% |
| **CSSV** | Does the §5.2 anti-discriminative negative result replicate? | **REPLICATES** — CSSV weight ratio ≈ 1.0, no discrimination |
| **Accuracy** | Is the proposed method non-inferior to FedDyn? (C4: margin 0.01) | **FAILS** — not established at any noise level; loses 8/10 seeds at 40% |

So: **the detector works, and the aggregation still does not beat FedDyn.** The
prior negative result (§5.1) now replicates on a fourth dataset and a second
modality. The ranking property the incentive mechanism depends on transfers
cleanly.

---

## 2. Setup

**Dataset.** `hopesb/student-depression-dataset` — 27,901 rows, 33 dropped for
missing/unmappable values, **27,868 usable**; binary `Depression` target at
**58.6% positive**. Split 80/20 stratified; scaler fit on the train split only.
Note: 58.6% is far above clinical prevalence (~15–35%), so this is a
self-selected survey sample, not an epidemiological one.

Dropped as degenerate: `Profession` (99.9% "Student"), `Work Pressure` and
`Job Satisfaction` (99.97% zero), `id`.

**Feature regime.** Two runs of every experiment were planned; the headline is
**`screening`**, which removes *"Have you ever had suicidal thoughts?"*. The
reason is measured, not assumed:

| Dataset | Regime | Centralized acc | AUC |
|---|---|---|---|
| student_depression (27.9k) | screening | 0.794 | 0.862 |
| student_depression (27.9k) | full | 0.839 | 0.912 |
| depression_student (502) | screening | 0.783 | 0.890 |
| depression_student (502) | **full** | **0.960** | **0.996** |

On the small 502-row set that one feature takes the task to **96% accuracy /
0.996 AUC** — fully saturated, and a saturated task cannot discriminate between
aggregators. Hence the screening regime for all FL work below.

**Protocol (frozen, identical for every method).** 10 clients, 5 active/round,
2 local epochs, 40 rounds, SGD lr=0.005 cosine-decayed, batch 128, Dirichlet
α=0.3, 20% per-client local test split, last-round metric, 10 seeds. Noise: 20%
or 40% of clients have 60% of training labels flipped; test labels stay clean.
Local LR was calibrated on **FedAvg only** and then frozen — no per-method tuning.
Per-client sizes range 136 / 1562 / 4563 (min/median/max), i.e. genuinely non-IID.

**Headline metric is balanced accuracy**, since a 58.6% base rate makes plain
accuracy reward majority-class prediction. Reference points: majority-class =
0.500 balanced acc; centralized pooled ceiling = 0.783.

---

## 3. The ranking test — this is the result that matters

Mean aggregation weight given to corrupted vs clean clients, pooled over all
rounds and seeds. Ratio < 1 = correctly down-weighted.

| Method | Signal | Ratio @20% | Ratio @40% | Probe-loss detection AUC |
|---|---|---|---|---|
| FedAvg / FedProx | size only | 0.961 | 0.971 | — |
| FedDyn | uniform | 1.000 | 1.000 | — |
| FedCE | gradient alignment | 0.988 | 1.033 | — |
| **Proposed (CSSV signal)** | ShapFed cosine | **0.994** | **1.013** | — |
| **Proposed (loss signal)** | loss detector | **0.740** | **0.783** | **0.917±0.063 / 0.874±0.088** |

Two findings:

1. **The loss detector transfers.** Ratio 0.74–0.78 here versus **0.69** on
   OCTMNIST — same direction, similar magnitude, in a different modality. The
   detection AUC of 0.874–0.917 is the cleaner number because it is computed
   directly from the probe losses and carries no dataset-size confound, unlike the
   weight ratio (the proposed weights blend a size prior at λ=0.5, and FedAvg's
   0.96–0.97 shows size alone has slight incidental correlation here). **This
   clears the pre-registered C3 bar of AUC ≥ 0.85 at both noise levels.**
2. **The ShapFed CSSV signal does not.** At ratio 0.994 / 1.013 it is
   non-discriminative — corrupted clients keep essentially full weight. This
   replicates the §5.2 negative result on tabular data, and confirms it is a
   property of cosine-of-update scoring under partial participation rather than an
   artefact of the imaging setup. Independent work (CoAst, SpectralFuse) reports
   the same class of weakness.

---

## 4. The accuracy test — fails again, as flagged

### 40% noisy clients

| Method | Balanced acc | Accuracy | Jain | Worst-client | Collapsed seeds |
|---|---|---|---|---|---|
| FedAvg | 0.649±0.116 | 0.663 | 0.816 | 0.333 | 3/10 |
| FedProx | 0.649±0.116 | 0.663 | 0.816 | 0.333 | 3/10 |
| **FedDyn** | **0.697±0.107** | 0.717 | 0.908 | 0.461 | 2/10 |
| FedCE | 0.657±0.129 | 0.671 | 0.815 | 0.369 | 4/10 |
| Proposed (CSSV) | 0.695±0.090 | 0.711 | 0.905 | 0.453 | 1/10 |
| Proposed (loss) | 0.665±0.113 | 0.693 | 0.867 | 0.399 | 2/10 |

### Seed-matched head-to-head vs FedDyn

| Noise | Proposed wins | Mean Δ | Median Δ | Non-inferiority (margin 0.01) |
|---|---|---|---|---|
| 0% | 1/10 | −0.0074 | −0.0040 | not established (p=0.40) |
| 20% | 6/10 | −0.0034 | +0.0023 | not established (p=0.32) |
| 40% | 1/10 | −0.0315 | −0.0188 | not established (p=0.66) |

**Verdict.** FedDyn is at least as good at every noise level. 20% is a genuine
tie (6/10 seed wins, median Δ positive); 0% and 40% are losses. **C4 is not
established anywhere**, so by our own pre-registered criteria the accuracy claim
fails on this dataset too. §5.1 replicates.

<b>What the proposed method does win:</b> it beats the size-weighted family
decisively under noise — at 40%, +0.016 balanced accuracy over FedAvg with
Jain 0.867 vs 0.816 and worst-client 0.399 vs 0.333; at 20%, +0.037 balanced
accuracy with markedly lower variance (±0.028 vs ±0.093) and 0/10 collapsed seeds
vs 1/10. The competitor it cannot beat is specifically FedDyn.

---

## 5. The sharpest new finding: accurate down-weighting *causes* the accuracy loss

Compare the two variants of the same method, which differ only in the quality
signal:

| Variant | Discriminates? | Balanced acc @40% |
|---|---|---|
| Proposed with **CSSV** signal | **No** (ratio 1.013) | **0.695** ≈ FedDyn (0.697) |
| Proposed with **loss** signal | **Yes** (ratio 0.783) | **0.665** |

The variant that fails to identify corrupted clients leaves weights near-uniform,
so it reduces to FedDyn and matches it. The variant that *successfully* identifies
them loses 0.030 balanced accuracy. **The better the detector works, the worse the
accuracy gets.**

This is direct causal evidence for the §5.1 diagnosis, and it is stronger than the
imaging study could provide: a 60%-label-flip client still carries partially
informative gradients, and FedDyn's dynamic regularization already absorbs them,
so suppressing that client discards more signal than noise. It also explains why
the accuracy claim has now failed on four datasets — the mechanism is not
dataset-specific, it is a property of the noise model.

**Implication for the incentive work:** the reward layer should consume the
detector's *ranking* while leaving aggregation weights alone. The
reputation-gated reward variant (§8.2), which touches payment only and not
aggregation, is the configuration these results support.

---

## 6. Limitations

- **Collapse-to-majority dominates the variance.** At 40% noise, 1–4 of 10 seeds
  land at balanced accuracy ≈ 0.50 (single-class prediction) for *every* method.
  This is a training-stability failure under a binary target with heavy label
  noise, not a subtle accuracy difference, and it inflates every std in §4. More
  rounds or a warmup may fix it; untested.
- **Ground-truth contribution not yet measured.** The ranking evidence here uses
  corrupted-vs-clean as a *binary* ground truth. Criterion **C1** requires a
  continuous leave-one-out marginal value and the reward-fidelity correlation
  against size-proportional and equal-split baselines. That needs a
  client-exclusion flag and is the immediate next step — **C1 is untested, not
  passed.**
- **No incentive module yet.** No reward rule, budget, or free-rider ladder
  (§9.3) has been run; C2 and C5 are untested. Free-riders should be easier to
  exclude than label-noise clients, but that is an expectation, not a result.
- One dataset, one architecture (2-layer MLP), one participation regime (N=10,
  m=5), and a single global feature scaler (mild centralisation).
- The 502-row set was used only for the saturation check, not the FL sweep — at
  50 rows/client it is too small for a 10-client partition with local test splits.

---

## 7. What this means for the proposal

The pivot in `docs/INCENTIVE_PROPOSAL.md` §5.3 rested on an empirical bet: that
the detector's *ranking* would transfer even though its *accuracy* benefit did
not. On this dataset the bet holds — detection AUC 0.87–0.92 with the accuracy
claim failing in the same runs. That is the cleanest possible demonstration that
the two claims are independent.

Two honest consequences:

1. The accuracy framing should be dropped entirely rather than defended.
   §10.3 committed us to reporting this replication, and it reproduced.
2. The incentive claim is **not yet demonstrated** — only its precondition is.
   C1 (reward fidelity vs a leave-one-out oracle) and C2 (free-rider starvation)
   remain the tests that decide the project.

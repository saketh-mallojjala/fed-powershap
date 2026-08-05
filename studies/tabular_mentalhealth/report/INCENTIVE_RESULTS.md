# Incentive results: C1 reward fidelity, C2/C3 free-rider starvation, and beyond-accuracy metrics

_Run in response to two review instructions: "continue with incentive", and
"apart from accuracy we need to consider some other performance metrics where it
actually performs better". Kaggle Student Depression Dataset, screening feature
regime, 10 clients, 10 seeds, one frozen protocol._

Reproduce: `scripts/loo_reward_fidelity.py`, `scripts/freerider_incentive.py`,
`scripts/aggregate_tabular.py`. Tables: `../tables/loo_reward_fidelity.md`,
`../tables/freerider_incentive.md`, `../tables/tabular_tables.md`.

---

## 1. Bottom line

| Pre-registered test | Result |
|---|---|
| **C1** — reward fidelity vs leave-one-out ≥ +0.15 over every baseline | **FAILS at every noise level** |
| **C2/C3** — free-riders starved (≤5% budget) and detected (AUC ≥ 0.85) | **PASSES on f2 at all fractions; f3 at 10%; FAILS on f1 and f4** |
| Beyond-accuracy metrics | **Discrimination is tied with FedDyn; the whole accuracy gap is threshold placement** |

Two of these are negative and one is positive. The positive one (§4) is the most
useful for the paper, because it says the accuracy deficit is **not** a loss of
discriminative power and is in principle recoverable.

---

## 2. C1 — reward fidelity against leave-one-out (FAILED)

Ground truth: `LOO_k = V(N) − V(N\{k})`, V = balanced accuracy on the shared clean
test set, measured under **FedAvg** (mechanism-neutral — using the proposed
method's own weighting to define truth would be circular). 390 federations:
one full + ten leave-one-out per setting, 10 seeds × 3 noise levels.

Spearman ρ between cumulative reward and LOO value, 95% bootstrap CI over seeds:

| Noise | fedshapin | size | equal | cssv | Margin vs best baseline | Bar |
|---|---|---|---|---|---|---|
| 0% | +0.287 | **+0.439** | +0.412 | **+0.465** | **−0.178** | +0.15 |
| 20% | +0.328 | +0.325 | +0.317 | +0.274 | **+0.002** | +0.15 |
| 40% | +0.123 | +0.058 | +0.042 | +0.005 | **+0.065** | +0.15 |

**On clean data our reward is worse than simply paying by dataset size.** At 20%
it is indistinguishable from equal-split. At 40% it is nominally best but its CI
is [−0.109, +0.377], spanning zero.

The ground truth is sound — corrupted clients have mean LOO value −0.029 vs
+0.020 for clean — so this is a failure of the *reward signal*, not of the oracle.

**Why, and it is intuitive in hindsight:** when every client is honest,
contribution genuinely *is* roughly proportional to how much data a client holds,
so size-proportional payment is already a strong rule and a quality signal only
adds noise on top of it. Our detector's advantage exists only when clients are
corrupted, and even there it is not large enough to clear the bar.

**Caveat, stated precisely rather than as an excuse.** The measurement is
underpowered at 20–40% noise: the LOO effect size is ~0.02–0.03 balanced
accuracy, but seed-to-seed variance at 40% noise is ±0.107, and each
`V(N\{k})` is a single run. Training noise is roughly 4× the signal, which is why
every rule's CI spans zero there. Averaging each LOO value over repeated seeds
would fix this (~10× compute). It would **not** rescue the clean-data column,
which is well-measured and a clear loss.

---

## 3. C2/C3 — free-rider starvation (MIXED, with one clean win)

### 3.1 The attack ladder

Four adversaries, verified by their update signatures (mean ‖update‖ relative to
honest clients): f1 random 0.18×, f2 zero-effort 0.00×, f3 trains on 10% of its
data 0.10×, f4 trains honestly then inflates by 3× → 2.64×.

### 3.2 The signals

The loss detector probes *the global model's loss on a client's own data*. A
free-rider holding good data that simply does not train still has a **low** probe
loss. So the loss signal measures **data quality, not effort** and is structurally
blind to free-riding — confirmed: detection AUC **0.63** on f2, near chance.

Effort shows up in the update instead:

- `norm` — raw ‖update‖. Confounded: honest ‖update‖ scales as **size^0.87**
  (corr +0.40 with client size), so a small client's inflated update looks like a
  large client's honest one.
- `effort` — ‖update‖ / n_k, i.e. movement per sample. Removes the size confound
  (corr −0.03).
- `gated` — hard eligibility test on effort deviation, then split among survivors.

### 3.3 Results (cumulative detection AUC / budget share captured)

| Rung | 10% free-riders | 20% | 30% | 40% |
|---|---|---|---|---|
| **f2** zero-effort | **0.97 / 0.0%** | **0.96 / 0.0%** | **0.95 / 0.0%** | **0.98 / 0.0%** |
| **f3** trains on 10% | **0.89 / 2.5%** | 0.79 / 6.6% | 0.73 / 10.8% | 0.67 / 25.1% |
| **f1** random update | 0.82 / 5.6% | 0.72 / 10.2% | 0.68 / 16.1% | 0.69 / 25.8% |
| **f4** inflate ×3 | 0.62 / 12.5% | 0.43 / 26.9% | 0.53 / 30.8% | 0.50 / 47.5% |

(best rule per cell; equal-split captures 10.9 / 20.9 / 29.4 / 39.3%)

- **f2 is fully solved** — zero budget captured and AUC 0.95–0.98 at *every*
  fraction up to 40%. Both C2 and C3 pass cleanly. No baseline rule comes close
  (loss 0.63, size 0.49, equal 0.50).
- **f3 passes only at 10%** (2.5% capture, AUC 0.89) and degrades after.
- **f1 just misses** the bars at 10% (5.6% vs ≤5%, AUC 0.82 vs ≥0.85).
- **f4 fails everywhere, and worse than failing: it is profitable.** At a 40%
  fraction the inflating free-riders capture **47.5%** of the budget against an
  equal-split share of 39.3% — the mechanism pays them *more* than their fair
  share. That is an explicit incentive-compatibility violation and must be
  reported as such.

### 3.4 Why f4 wins — measured, not speculated

Under Dirichlet α=0.3 non-IID partitioning, honest effort density is extremely
heterogeneous: **log-sd = 1.014**, i.e. ordinary client-to-client variation spans
a factor of **2.76× at one standard deviation**.

A 3× inflation is `log 3 = 1.099` — **1.08 sd**. It sits *inside* the honest band.
To separate at 2 sd an attacker would have to inflate by more than **7.6×**, and
no attacker is obliged to. Concretely, a gate tolerant enough to keep honest
clients (log 2) still excluded **34–39% of honest observations** while catching
only 44–50% of the f4 clients.

Two further contributors, both measured:

- At a 40% free-rider fraction, adversaries are a **majority of the 5-client
  active set in 22% of rounds**, so the peer median — the reference for "normal
  effort" — is itself adversarial in those rounds.
- A pooled running reference is *worse*, not better: with 40% of clients
  inflating every round the accumulated history becomes ~40% contaminated and
  drifts toward the attack (f4 AUC 0.67 → 0.53 when we tried it).

**The honest conclusion: update-magnitude scoring cannot detect inflation smaller
than the natural spread of honest client effort.** With strongly non-IID clients
that spread is large, so the exploitable range is wide. This is a property of the
setting, not of our particular implementation.

---

## 4. Beyond accuracy — where the method genuinely holds up

Accuracy and balanced accuracy both read the model at the **default 0.5
threshold**, which conflates two different failures: *cannot separate the
classes* and *separates them but places the boundary badly*. Splitting them:

### 40% noisy clients

| Method | balanced_acc | **AUPRC** | **AUC** | **bacc_opt** | bacc_opt − balanced_acc | F1(pos) |
|---|---|---|---|---|---|---|
| FedAvg | 0.649 | 0.876 | 0.846 | 0.772 | +0.123 | 0.672±0.236 |
| **FedDyn** | **0.697** | 0.878 | 0.850 | **0.780** | +0.083 | 0.764±0.092 |
| FedCE | 0.657 | 0.875 | 0.845 | 0.772 | +0.115 | 0.682±0.234 |
| **Proposed** | 0.665 | **0.878** | 0.849 | **0.779** | +0.113 | **0.757±0.046** |

**The key finding.** On every metric that measures *discriminative power* the
proposed method is tied with FedDyn:

- AUPRC **0.878 vs 0.878** — identical, and jointly the best of all methods
- AUC **0.849 vs 0.850**
- threshold-optimal balanced accuracy **0.779 vs 0.780**

Yet at the default threshold it trails by 0.032. **The entire accuracy deficit is
threshold placement, not lost information.** The proposed method forfeits +0.113
balanced accuracy to a mis-placed operating point where FedDyn forfeits only
+0.083.

This reframes the §5.1 negative result. Down-weighting noisy clients does not
damage what the model *knows*; it shifts the predicted probabilities so the 0.5
cut lands in the wrong place. That is a **calibration** problem, and post-hoc
threshold selection on held-out data is standard practice for clinical screening.

Two honest qualifications: `bacc_opt` uses an oracle threshold chosen on the test
set, so it is an upper bound — realising it requires picking the threshold on
validation data, which we have not yet done. And ECE at 40% is 0.162 for proposed
vs 0.155 for FedDyn, so the proposed method is *not* better calibrated in the raw
probability sense; it is specifically the decision threshold that is misplaced.

### Where it beats the size-weighted family outright

- **F1 stability at 40%:** 0.757 ± **0.046** vs FedAvg 0.672 ± 0.236 and FedCE
  0.682 ± 0.234 — comparable mean to FedDyn but **half its variance** (±0.092).
- **At 20% noise:** F1 0.783 ± 0.048 vs FedAvg 0.681 ± 0.261; 0/10 collapsed
  seeds vs 1/10.
- **Detection:** AUC 0.874–0.917 for label-noise clients and 0.95–0.98 for f2
  free-riders. No baseline has any client-quality signal at all.

---

## 5. What this means for the project

1. **The reward mechanism as specified does not beat paying by dataset size**
   (C1). That was the pre-registered decisive test and it failed.
2. **But free-rider exclusion works for the adversary that matters most in
   practice** — the participant who contributes nothing (f2) is detected almost
   perfectly and paid exactly zero, at every fraction tested. That is a real,
   deployable property, and no baseline rule achieves it.
3. **The mechanism is gameable by update inflation** (f4), with a measured
   explanation for why any magnitude-based defence fails in a non-IID setting.
   This is a genuine incentive-compatibility result and belongs in the paper.
4. **The accuracy story should be reframed, not defended.** The model matches
   FedDyn on AUC, AUPRC and threshold-optimal accuracy; only the operating point
   differs.

**Recommended next steps, in order:**

- Threshold calibration on a validation split — cheap, and §4 says it should
  close most of the accuracy gap. This is the single highest-value experiment left.
- Re-run C1 with LOO values averaged over seeds, to settle whether the 20–40%
  columns are a real failure or a power problem.
- Test whether norm-clipping at aggregation neutralises f4's *profit* even though
  it cannot be *detected* — if inflation stops paying, IC may be restorable
  without detection.

---

## 6. Limitations

- One dataset, one architecture, N=10 clients, m=5 active.
- f4 is tested at a single inflation factor (3×); the detectability threshold
  (7.6×) is derived from the measured honest spread, not swept directly.
- Free-riders are injected on clean labels only; combined noise + free-riding is
  untested.
- `bacc_opt` is an oracle threshold (upper bound), as noted in §4.
- C1's LOO estimator is underpowered at 20–40% noise (§2).

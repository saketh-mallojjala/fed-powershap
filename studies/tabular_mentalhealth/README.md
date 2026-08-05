# Study: tabular mental-health case study (FedShapIn precursor)

Self-contained results folder for the depression-screening experiments run on
2026-07-28, in response to the review instruction *"try with a small Kaggle mental
dataset. Share the results then will see."*

Everything this study produced lives here. The code lives in `scripts/` and the
main package (see [Reproducing](#reproducing)); nothing here is generated at
import time, so the folder can be copied or shared on its own.

## Contents

| Path | What it is |
|---|---|
| `report/TABULAR_RESULTS.md` | **Start here.** Setup, the accuracy/detection findings, limitations. |
| `report/INCENTIVE_RESULTS.md` | The incentive work: C1 reward fidelity, C2/C3 free-rider starvation, beyond-accuracy metrics. |
| `tables/tabular_tables.md` | Generated results tables (all methods × noise × seeds, mechanism test, non-inferiority). |
| `tables/tabular_ceiling.json` | Centralized pooled-data ceiling per dataset × feature regime. |
| `tables/loo_reward_fidelity.md` | C1: reward vs leave-one-out ground truth. |
| `tables/freerider_incentive.md` | C2/C3: free-rider budget capture and detection, per rung × fraction. |
| `tables/dcfi_results.md` | DCFI-UPA: 8 attacks × 2 methods × 8 seeds, all 15 metrics. |
| `logs/` | 170 raw run logs (JSONL, one per method × noise × seed). Per-round weights, active sets, probe losses, update norms, corrupted-client ground truth. |
| `logs_loo/` | 390 leave-one-out runs backing C1. |
| `logs_freerider/` | 160 free-rider runs backing C2/C3. |
| `logs_dcfi/` | 144 DCFI-UPA runs (8 attacks × 2 methods × 8 seeds). |
| `logs_dev/` | Scaffolding runs kept for provenance: the FedAvg-only LR calibration (4 runs) and one pipeline smoke test. Not part of any reported result. |

## Headline

| Pre-registered test | Result |
|---|---|
| Ranking — does the loss detector separate corrupted from clean clients? (C3: AUC ≥ 0.85) | **PASSES** — 0.917 @20% noise, 0.874 @40% |
| CSSV — does the anti-discriminative negative result replicate? | **REPLICATES** — weight ratio ≈ 1.0 |
| Accuracy — is the proposed method non-inferior to FedDyn? (C4: margin 0.01) | **FAILS** — loses 8/10 seed-matched runs at 40% |
| C1 — reward fidelity beats size-proportional by ≥0.15? | **FAILS** at every noise level |
| C2/C3 — free-riders starved and detected? | **PASSES on f2** (0% budget, AUC 0.95–0.98, all fractions); fails on f4 |
| Beyond accuracy — AUPRC / AUC / threshold-optimal balanced acc | **TIED with FedDyn** — the gap is threshold placement, not discrimination |
| DCFI-UPA reward channel — IR, budget balance, fairness indices | **PASSES** — IR 98–100%, reward-to-cost 3.8–4.6, Jain 0.85–0.95 |
| DCFI-UPA safety gate — accuracy under 8 attacks | **FAILS** — loses on 8 of 9 conditions; 3/8 seeds collapse with no adversaries |

The detector's ranking transfers to tabular data; the aggregation still does not
beat FedDyn. See `report/TABULAR_RESULTS.md` §5 for the causal explanation
(better detection → worse accuracy).

The incentive layer is now tested too. `report/INCENTIVE_RESULTS.md` covers it:
reward fidelity loses to paying by dataset size (C1, §2); zero-effort free-riders
are excluded perfectly but update-inflation is *profitable* against us (C2/C3,
§3); and the accuracy gap turns out to be threshold placement rather than lost
discriminative power (§4). Still untested: C5 budget balance under adversaries,
and combined label-noise + free-riding.

## Data

Not vendored here (Kaggle-licensed). Download into `data_cache/mentalhealth/`:

```bash
export KAGGLE_CONFIG_DIR=$(pwd)
kaggle datasets download -d hopesb/student-depression-dataset \
  -p data_cache/mentalhealth --unzip          # 27,901 rows -- primary
kaggle datasets download -d ikynahidwin/depression-student-dataset \
  -p data_cache/mentalhealth --unzip          # 502 rows -- saturation check only
```

Note the `adilshamim8/student-depression-dataset` ref cited in earlier drafts is
no longer live; `hopesb` is the same data.

## Reproducing

Run from the repository root. Total wall time ≈ 15 min on CPU.

```bash
# 1. Centralized ceiling FIRST -- confirms there is discriminative headroom
#    before any federated run (and shows the "full" regime is saturated).
python3 scripts/tabular_ceiling.py

# 2. The sweep: 5 methods x 3 noise levels x 10 seeds, one frozen protocol.
python3 scripts/run_tabular_sweep.py --seeds 0 1 2 3 4 5 6 7 8 9

# 3. The CSSV-signal ablation (the §5.2 negative-result check).
for n in 0.2 0.4; do for s in 0 1 2 3 4 5 6 7 8 9; do
  nn=$(printf '%02d' $(python3 -c "print(round($n*100))"))
  python3 main.py --method proposed --contrib-signal cssv \
    --dataset student_depression --seed $s \
    --exp-name student_depression_screening_proposedcssv_n${nn}_seed${s} \
    --device cpu --log-dir studies/tabular_mentalhealth/logs \
    --feature-regime screening --noisy-client-frac $n --label-noise-rate 0.6 \
    --num-clients 10 --candidate-size-d 10 --active-size-m 5 --num-rounds 40 \
    --local-epochs 2 --local-lr 0.005 --batch-size 128 --alpha 0.3 \
    --local-test-frac 0.2 --eval-every 2 --lr-schedule cosine --num-workers 0 \
    --hidden-dim 64 --mlp-depth 2 --mlp-dropout 0.1
done; done

# 4. Tables + mechanism analysis + the beyond-accuracy metrics.
python3 scripts/aggregate_tabular.py

# 5. C1 -- reward fidelity vs a leave-one-out ground truth (390 federations, ~35 min).
python3 scripts/loo_reward_fidelity.py --seeds 0 1 2 3 4 5 6 7 8 9 --noises 0.0 0.2 0.4

# 6. C2/C3 -- free-rider ladder x fraction breakdown curve (160 runs, ~15 min).
python3 scripts/freerider_incentive.py --seeds 0 1 2 3 4 5 6 7 8 9 \
    --modes f1 f2 f3 f4 --fracs 0.1 0.2 0.3 0.4
```

The sweep is resumable — a run whose log already exists is skipped unless
`--force`, so step 2 can be re-invoked with more seeds without redoing work.

## Protocol (frozen)

10 clients, 5 active/round, 2 local epochs, 40 rounds, SGD lr=0.005
cosine-decayed, batch 128, Dirichlet α=0.3, 20% per-client local test split,
last-round metric. Noise: 20%/40% of clients have 60% of training labels flipped;
test labels stay clean. **Local LR was calibrated on FedAvg alone and then fixed
for every method** — no per-method tuning. Headline metric is balanced accuracy
(the target is 58.6% positive).

## Code added for this study

| File | Purpose |
|---|---|
| `data/tabular.py` | Kaggle CSV loaders, encoding, `screening`/`full` feature regimes, leakage-safe scaling |
| `models/mlp.py` | `TabularMLP` with a `classifier` head so CSSV keys on it |
| `scripts/tabular_ceiling.py` | Centralized ceiling |
| `scripts/run_tabular_sweep.py` | Frozen-protocol sweep driver |
| `scripts/aggregate_tabular.py` | Tables, mechanism/ranking test, non-inferiority, seed-matched comparison, beyond-accuracy metrics |
| `scripts/loo_reward_fidelity.py` | C1: leave-one-out ground truth + reward-rule comparison |
| `scripts/freerider_incentive.py` | C2/C3: attack ladder analysis, signal comparison, breakdown curve |
| `federated/client.py` | Free-rider behaviours f1–f4; `local_train` refactored to `_dispatch_solver` (verified bit-identical on existing runs) |
| `data/dataset.py` | Free-rider client assignment, disjoint from the label-noise clients |
| `config.py` | `freerider_*` knobs; `exclude_client` for leave-one-out |
| `main.py` | Applies `--exclude-client` after partitioning so LOO is a true marginal value |
| `federated/server.py` | `balanced_acc` + binary `auc`; calibration/ranking metrics (`ece`, `brier`, `auprc`, `f1_pos`, `bacc_opt`, all validated against sklearn); per-round update norms and CSSV logged as diagnostics |

## Related documents

- **`docs/DCFI_TECHNICAL_REPORT.md` / `.pdf` — the full DCFI-UPA technical report:
  specification, all 15 metrics, the 8-attack evaluation, design corrections and
  the honest verdict. Start here for the framework.**
- `docs/INCENTIVE_PROPOSAL.md` — the proposal these tests were pre-registered in (§5 prior negative result, §10 criteria)
- `docs/SYSTEM_OVERVIEW.md` §7 — the earlier medical-imaging results this replicates
- `docs/ROBUST_SHAPFED_RESULTS.md` §2.2 — the original CSSV negative result

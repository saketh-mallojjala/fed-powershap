# Revision evidence, organised by reviewer comment

Each table is named for the review point it answers.

| file prefix | reviewer comment |
|:--|:--|
| `A2_` | A: how is corrupted data distinguished from difficult / shifted but correctly labelled data? |
| `A3_` | A: noise levels, non-IID degrees, participation ratios, mechanism results on all datasets |
| `B2_` | B: corruption levels and noise types |
| `B3_` | B: model poisoning |
| `B5_` | B: scaling beyond 10 clients |

## How to read the main table

`paired_delta` is the per-seed difference against FedDyn, averaged. Both
methods see the identical partition and identical corrupted clients in a
given seed, so the paired difference is the correct statistic: its spread
is ~0.007 against ~0.045 for either arm alone. `sign` reports whether every
seed agreed; `all+` with a positive delta is the claim, `mixed` is not.

`weight_ratio_*` is the mean aggregation weight a ground-truth group
received relative to clean clients, computed within each round and then
averaged. Below 1 means down-weighted; 1.0 means treated identically.

## Caveats carried from the runs

- The three noise-aware baselines are mechanism-faithful reimplementations,
  not ports; see `federated/noise_baselines.py` for exactly what differs.
- OCTMNIST is subsampled to 20,000 training images; at full size one
  40-round run costs 6.3 hours.
- `n@` columns give the seed count behind each mean.

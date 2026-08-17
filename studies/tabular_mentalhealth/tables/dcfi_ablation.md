# DCFI-UPA — reward-formula ablations and parameter sensitivity

Recomputed from 40 logged runs (the reward rule is a pure function of per-round signals already recorded, so no retraining is needed). Every variant spends the same budget; only the shape of the split changes.


## A. Does each term in the reward formula earn its place?

Averaged over all attack scenarios. `adv share` = fraction of budget captured by adversaries (lower better; equal-split gives them 40%). `IR` = share of honest clients paid above cost.

| Reward formula | Adv share | Fidelity ρ | Reward/cost | IR | Jain(reward) | ρ(reward,size) |
|---|---|---|---|---|---|---|
| full (0.2/0.5/0.3) | 3.90% | +0.145 | 4.4 | 100% | 0.877 | +0.215 |
| no base term | 3.89% | +0.136 | 4.4 | 100% | 0.849 | +0.211 |
| no utility term | 3.94% | +0.045 | 4.4 | 100% | 0.867 | +0.122 |
| no fairness term | 4.00% | +0.182 | 4.4 | 100% | 0.907 | +0.200 |
| utility only | 3.97% | +0.210 | 4.4 | 100% | 0.873 | +0.192 |
| fairness only | 3.83% | +0.033 | 4.4 | 98% | 0.799 | +0.068 |
| equal split (base only) | 4.14% | +0.000 | 4.3 | 100% | 0.938 | +0.072 |

## B. Base-payment weight (individual rationality vs efficiency)

The base term pays every participant regardless of measured utility. Raising it protects individual rationality; lowering it concentrates reward on high-utility clients but also on anyone who fakes utility.

| w_base | Adv share | Fidelity ρ | Reward/cost | IR | Jain(reward) |
|---|---|---|---|---|---|
| 0.0 | 3.89% | +0.136 | 4.4 | 100% | 0.849 |
| 0.1 | 3.89% | +0.143 | 4.4 | 100% | 0.868 |
| 0.2 | 3.90% | +0.145 | 4.4 | 100% | 0.877 |
| 0.3 | 3.92% | +0.146 | 4.4 | 100% | 0.885 |
| 0.5 | 3.96% | +0.141 | 4.4 | 100% | 0.902 |

## C. Reward cap (concentration control)

Water-filling cap on any single client's share of a round's budget. 1.0 disables it.

| cap | Adv share | Fidelity ρ | Reward/cost | IR | Jain(reward) |
|---|---|---|---|---|---|
| 0.2 | 3.71% | +0.153 | 4.4 | 100% | 0.798 |
| 0.3 | 3.93% | +0.131 | 4.4 | 100% | 0.907 |
| 0.4 | 3.90% | +0.145 | 4.4 | 100% | 0.877 |
| 0.6 | 3.80% | +0.157 | 4.4 | 100% | 0.831 |
| 1.0 | 3.71% | +0.153 | 4.4 | 100% | 0.798 |

## D. Holdback fraction (deferred settlement)

Fraction withheld each round and released only once the client's reliability clears the threshold.

| holdback | Adv share | Fidelity ρ | Reward/cost | IR | Jain(reward) |
|---|---|---|---|---|---|
| 0.0 | 4.48% | +0.158 | 4.5 | 100% | 0.882 |
| 0.1 | 4.20% | +0.152 | 4.4 | 100% | 0.880 |
| 0.2 | 3.90% | +0.145 | 4.4 | 100% | 0.877 |
| 0.3 | 3.60% | +0.145 | 4.3 | 100% | 0.874 |
| 0.5 | 2.97% | +0.126 | 4.2 | 100% | 0.868 |

## E. Utility/fairness balance (efficiency vs redistribution)

Splitting the non-base budget between measured utility and minority-class fairness contribution.

| w_utility (rest to fairness) | Adv share | Fidelity ρ | Reward/cost | IR | Jain(reward) |
|---|---|---|---|---|---|
| 0.0 | 3.91% | +0.042 | 4.4 | 100% | 0.846 |
| 0.2 | 3.90% | +0.099 | 4.4 | 100% | 0.861 |
| 0.4 | 3.89% | +0.125 | 4.4 | 100% | 0.869 |
| 0.6 | 3.90% | +0.145 | 4.4 | 100% | 0.876 |
| 0.8 | 3.93% | +0.166 | 4.4 | 100% | 0.886 |

## E2. Participation cost (where does individual rationality break?)

Cost is expressed as a share of the per-round budget, charged per round a client is selected. IR = fraction of honest clients whose cumulative reward covers their cumulative cost.

| cost/round | Reward-to-cost | IR (honest) | IR (w_base=0) |
|---|---|---|---|
| 0.05 | 4.36 | 100% | 100% |
| 0.10 | 2.18 | 99% | 96% |
| 0.15 | 1.45 | 84% | 79% |
| 0.20 | 1.09 | 55% | 53% |
| 0.25 | 0.87 | 31% | 32% |
| 0.30 | 0.73 | 14% | 19% |

## F. Chosen formula, per scenario

| Scenario | Adv share | Fidelity ρ | Reward/cost | IR | Jain(reward) | ρ(reward,size) |
|---|---|---|---|---|---|---|
| none | 0.00% | +0.165 | 3.8 | 100% | 0.871 | +0.363 |
| zero_update | 0.82% | +0.245 | 4.5 | 100% | 0.863 | +0.200 |
| copied_update | 3.33% | +0.158 | 4.4 | 100% | 0.845 | +0.161 |
| scaled_global | 6.25% | -0.113 | 4.5 | 100% | 0.914 | +0.173 |
| random_update | 9.12% | +0.270 | 4.6 | 100% | 0.892 | +0.178 |

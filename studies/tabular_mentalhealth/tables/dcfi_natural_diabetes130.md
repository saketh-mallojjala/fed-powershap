# DCFI-UPA results — dual-channel fair incentive & utility-preserving aggregation

Tabular mental-health protocol, 5 seeds, adversary fraction 40%. `none` = no adversaries.

## Performance channel

| Attack | Method | Balanced acc | AUC | F1(pos) | Worst-client | Conv. round | Collapsed |
|---|---|---|---|---|---|---|---|
| none | feddyn | 0.622±0.002 | 0.678±0.003 | 0.558±0.009 | 0.567±0.020 | 4 | 0/5 |
| none | dcfi | 0.621±0.003 | 0.673±0.005 | 0.564±0.008 | 0.557±0.036 | 7 | 0/5 |
| zero_update | feddyn | 0.617±0.005 | 0.672±0.005 | 0.552±0.025 | 0.559±0.025 | 5 | 0/5 |
| zero_update | dcfi | 0.616±0.008 | 0.671±0.007 | 0.545±0.035 | 0.559±0.033 | 8 | 0/5 |
| copied_update | feddyn | 0.608±0.017 | 0.671±0.011 | 0.500±0.067 | 0.553±0.022 | 4 | 0/5 |
| copied_update | dcfi | 0.616±0.003 | 0.672±0.005 | 0.544±0.020 | 0.548±0.035 | 6 | 0/5 |
| random_update | feddyn | 0.618±0.004 | 0.673±0.004 | 0.555±0.021 | 0.553±0.031 | 5 | 0/5 |
| random_update | dcfi | 0.612±0.008 | 0.670±0.006 | 0.532±0.039 | 0.547±0.041 | 7 | 0/5 |
| label_flip | feddyn | 0.600±0.029 | 0.645±0.038 | 0.546±0.048 | 0.523±0.022 | 5 | 0/5 |
| label_flip | dcfi | 0.598±0.020 | 0.647±0.023 | 0.517±0.082 | 0.533±0.037 | 5 | 0/5 |

## Reward channel (DCFI only)

| Attack | Adversary reward share | ρ(reward, LOO) | MAE | Reward-to-cost | IR rate | Jain(useful rewards) | Jain(selection) | ρ(reward, size) |
|---|---|---|---|---|---|---|---|---|
| none | 0.00% | 0.874±0.051 | 0.040±0.011 | 3.8±0.0 | 100% | 0.802±0.017 | 0.961±0.015 | 0.235±0.073 |
| zero_update | 0.00% | 0.476±0.179 | 0.052±0.011 | 4.6±0.1 | 100% | 0.892±0.031 | 0.817±0.006 | 0.215±0.126 |
| copied_update | 2.45% | 0.500±0.232 | 0.067±0.022 | 4.6±0.1 | 100% | 0.878±0.069 | 0.842±0.012 | 0.155±0.156 |
| random_update | 3.18% | 0.445±0.130 | 0.053±0.008 | 4.6±0.1 | 100% | 0.878±0.032 | 0.850±0.013 | 0.190±0.162 |
| label_flip | 45.42% | 0.682±0.130 | 0.075±0.016 | 3.5±0.1 | 100% | 0.817±0.030 | 0.961±0.017 | 0.162±0.189 |

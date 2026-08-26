# DCFI-UPA results — dual-channel fair incentive & utility-preserving aggregation

Tabular mental-health protocol, 5 seeds, adversary fraction 40%. `none` = no adversaries.

## Performance channel

| Attack | Method | Balanced acc | AUC | F1(pos) | Worst-client | Conv. round | Collapsed |
|---|---|---|---|---|---|---|---|
| none | feddyn | 0.848±0.015 | 0.905±0.013 | 0.865±0.017 | 0.755±0.041 | 17 | 0/5 |
| none | dcfi | 0.832±0.013 | 0.903±0.010 | 0.849±0.017 | 0.732±0.065 | 14 | 0/5 |
| zero_update | feddyn | 0.793±0.015 | 0.872±0.022 | 0.802±0.020 | 0.512±0.060 | 21 | 0/5 |
| zero_update | dcfi | 0.785±0.018 | 0.872±0.019 | 0.798±0.009 | 0.505±0.056 | 17 | 0/5 |
| copied_update | feddyn | 0.815±0.018 | 0.891±0.021 | 0.821±0.027 | 0.618±0.113 | 14 | 0/5 |
| copied_update | dcfi | 0.797±0.023 | 0.880±0.016 | 0.807±0.018 | 0.570±0.041 | 10 | 0/5 |
| random_update | feddyn | 0.719±0.038 | 0.794±0.054 | 0.720±0.053 | 0.465±0.109 | 17 | 0/5 |
| random_update | dcfi | 0.791±0.021 | 0.866±0.016 | 0.799±0.023 | 0.556±0.071 | 18 | 0/5 |
| label_flip | feddyn | 0.776±0.029 | 0.851±0.036 | 0.772±0.047 | 0.463±0.096 | 21 | 0/5 |
| label_flip | dcfi | 0.783±0.013 | 0.860±0.015 | 0.786±0.027 | 0.451±0.104 | 21 | 0/5 |

## Reward channel (DCFI only)

| Attack | Adversary reward share | ρ(reward, LOO) | MAE | Reward-to-cost | IR rate | Jain(useful rewards) | Jain(selection) | ρ(reward, size) |
|---|---|---|---|---|---|---|---|---|
| none | 0.00% | 0.720±0.204 | 0.086±0.049 | 5.5±0.2 | 100% | 0.876±0.062 | 0.960±0.032 | 0.647±0.141 |
| zero_update | 0.00% | 0.689±0.518 | 0.132±0.091 | 9.9±0.0 | 100% | 0.992±0.011 | 0.692±0.000 | 0.707±0.272 |
| copied_update | 1.00% | 0.617±0.275 | 0.156±0.038 | 9.8±0.0 | 100% | 0.988±0.022 | 0.714±0.000 | 0.707±0.277 |
| random_update | 0.86% | 0.813±0.259 | 0.124±0.052 | 9.7±0.2 | 100% | 0.987±0.020 | 0.716±0.004 | 0.707±0.274 |
| label_flip | 41.86% | 0.840±0.150 | 0.282±0.253 | 5.8±0.6 | 100% | 0.994±0.009 | 0.987±0.012 | 0.236±0.607 |

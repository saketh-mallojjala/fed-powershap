# C1 — reward fidelity vs a leave-one-out ground truth

Spearman ρ between cumulative reward and LOO marginal value (balanced accuracy, N=16 clients), over 5 seeds. Ground truth measured under FedAvg (mechanism-neutral). 95% CI by bootstrap over seeds (10k resamples).

**Pre-registered bar (§10.1 C1):** fedshapin must beat *each* of `size`, `equal`, `cssv` by ≥ 0.15 with non-overlapping CIs.


### Clean (0% noisy clients)

| Reward rule | Spearman ρ | 95% CI | Δ vs fedshapin |
|---|---|---|---|
| **fedshapin** | +0.086 | [-0.199, +0.416] | — |
| quality | +0.095 | [-0.187, +0.425] | -0.009 |
| cssv | +0.091 | [-0.183, +0.408] | -0.005 |
| size | +0.107 | [-0.094, +0.391] | -0.021 |
| equal | +0.097 | [-0.181, +0.433] | -0.011 |

**C1 at this noise level: FAIL** — smallest margin over a baseline is -0.021 (bar: +0.150).


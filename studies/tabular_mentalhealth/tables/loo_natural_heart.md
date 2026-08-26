# C1 — reward fidelity vs a leave-one-out ground truth

Spearman ρ between cumulative reward and LOO marginal value (balanced accuracy, N=4 clients), over 5 seeds. Ground truth measured under FedAvg (mechanism-neutral). 95% CI by bootstrap over seeds (10k resamples).

**Pre-registered bar (§10.1 C1):** fedshapin must beat *each* of `size`, `equal`, `cssv` by ≥ 0.15 with non-overlapping CIs.


### Clean (0% noisy clients)

| Reward rule | Spearman ρ | 95% CI | Δ vs fedshapin |
|---|---|---|---|
| **fedshapin** | -0.040 | [-0.560, +0.600] | — |
| quality | -0.040 | [-0.560, +0.600] | +0.000 |
| cssv | -0.160 | [-0.520, +0.200] | +0.120 |
| size | +0.200 | [-0.360, +0.720] | -0.240 |

**C1 at this noise level: FAIL** — smallest margin over a baseline is -0.240 (bar: +0.150).


# C1 — reward fidelity vs a leave-one-out ground truth

Spearman ρ between cumulative reward and LOO marginal value (balanced accuracy, N=10 clients), over 10 seeds. Ground truth measured under FedAvg (mechanism-neutral). 95% CI by bootstrap over seeds (10k resamples).

**Pre-registered bar (§10.1 C1):** fedshapin must beat *each* of `size`, `equal`, `cssv` by ≥ 0.15 with non-overlapping CIs.


### Clean (0% noisy clients)

| Reward rule | Spearman ρ | 95% CI | Δ vs fedshapin |
|---|---|---|---|
| **fedshapin** | +0.287 | [+0.064, +0.510] | — |
| quality | +0.274 | [+0.045, +0.501] | +0.013 |
| cssv | +0.465 | [+0.234, +0.669] | -0.178 |
| size | +0.439 | [+0.176, +0.678] | -0.152 |
| equal | +0.412 | [+0.162, +0.646] | -0.124 |

**C1 at this noise level: FAIL** — smallest margin over a baseline is -0.178 (bar: +0.150).


### 20% noisy clients

| Reward rule | Spearman ρ | 95% CI | Δ vs fedshapin |
|---|---|---|---|
| **fedshapin** | +0.328 | [+0.085, +0.566] | — |
| quality | +0.330 | [+0.090, +0.566] | -0.002 |
| cssv | +0.274 | [+0.072, +0.475] | +0.053 |
| size | +0.325 | [+0.071, +0.590] | +0.002 |
| equal | +0.317 | [+0.072, +0.573] | +0.011 |

**C1 at this noise level: FAIL** — smallest margin over a baseline is +0.002 (bar: +0.150).

Ground-truth sanity check: mean LOO value of corrupted clients = -0.0294 vs clean = +0.0197.


### 40% noisy clients

| Reward rule | Spearman ρ | 95% CI | Δ vs fedshapin |
|---|---|---|---|
| **fedshapin** | +0.123 | [-0.109, +0.377] | — |
| quality | +0.119 | [-0.111, +0.371] | +0.004 |
| cssv | +0.005 | [-0.235, +0.249] | +0.118 |
| size | +0.058 | [-0.158, +0.293] | +0.065 |
| equal | +0.042 | [-0.175, +0.265] | +0.081 |

**C1 at this noise level: FAIL** — smallest margin over a baseline is +0.065 (bar: +0.150).

Ground-truth sanity check: mean LOO value of corrupted clients = -0.0256 vs clean = +0.0146.


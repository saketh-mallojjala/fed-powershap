# A. Dirichlet-alpha sensitivity

8 seeds, student_depression, everything except alpha held fixed. Small alpha = extreme label skew across clients; alpha=10 is effectively IID.

## A1. Does the proposed system hold up as heterogeneity varies?

| alpha | Attack | FedDyn bal-acc | DCFI bal-acc | Δ | FedDyn collapsed | DCFI collapsed |
|---|---|---|---|---|---|---|
| 0.1 | none | 0.750±0.039 | 0.726±0.094 | -0.024 | 0/8 | 1/8 |
| 0.1 | zero_update | 0.708±0.096 | 0.686±0.111 | -0.022 | 1/8 | 2/8 |
| 0.3 | none | 0.773±0.024 | 0.760±0.032 | -0.013 | 0/8 | 0/8 |
| 0.3 | zero_update | 0.749±0.037 | 0.770±0.020 | +0.021 | 0/8 | 0/8 |
| 0.5 | none | 0.778±0.009 | 0.783±0.011 | +0.004 | 0/8 | 0/8 |
| 0.5 | zero_update | 0.764±0.011 | 0.775±0.014 | +0.011 | 0/8 | 0/8 |
| 1.0 | none | 0.785±0.005 | 0.784±0.017 | -0.001 | 0/8 | 0/8 |
| 1.0 | zero_update | 0.767±0.022 | 0.755±0.052 | -0.012 | 0/8 | 0/8 |
| 10.0 | none | 0.786±0.004 | 0.788±0.003 | +0.001 | 0/8 | 0/8 |
| 10.0 | zero_update | 0.779±0.007 | 0.784±0.005 | +0.005 | 0/8 | 0/8 |

## A2. Does reward fidelity track client heterogeneity?

The hypothesis from the natural-partition runs: a reward rule can only rank clients that genuinely differ. Lower alpha => more different clients => fidelity should rise. Reference points: synthetic alpha=0.3 gave rho 0.165, real hospital/specialty partitions gave 0.72-0.87.

| alpha | ρ(reward, LOO) | Adversary reward share | IR | Jain(reward) | Jain(selection) |
|---|---|---|---|---|---|
| 0.1 | 0.027±0.436 | 0.00% | 100% | 0.888±0.028 | 0.970±0.018 |
| 0.3 | 0.295±0.407 | 0.00% | 100% | 0.881±0.029 | 0.976±0.010 |
| 0.5 | 0.520±0.275 | 0.00% | 100% | 0.868±0.048 | 0.979±0.009 |
| 1.0 | 0.617±0.264 | 0.00% | 100% | 0.855±0.035 | 0.974±0.007 |
| 10.0 | 0.715±0.118 | 0.00% | 100% | 0.859±0.067 | 0.976±0.008 |

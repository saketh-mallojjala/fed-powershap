# Tabular mental-health results — student_depression (screening regime)

Last-round metrics, mean±std over 10 seeds. Frozen protocol: 10 clients, 5 active/round, 2 local epochs, 40 rounds, SGD lr=0.005 cosine-decayed, Dirichlet α=0.3. **Balanced accuracy is the headline** (target is 58.6% positive, so plain accuracy flatters majority-class prediction).

Reference points: majority-class = 0.586 acc / 0.500 balanced acc; centralized pooled-data ceiling = 0.794 acc / 0.783 balanced acc.


### Clean (0% noisy clients)

| Method | Balanced acc | Accuracy | AUC | Jain | Worst-client | Collapsed seeds |
|---|---|---|---|---|---|---|
| fedavg | 0.757±0.035 | 0.778±0.020 | 0.870±0.003 | 0.961±0.042 | 0.595±0.128 | 0/10 |
| fedprox | 0.757±0.035 | 0.778±0.019 | 0.870±0.003 | 0.961±0.042 | 0.595±0.128 | 0/10 |
| feddyn | 0.772±0.023 | 0.776±0.035 | 0.868±0.003 | 0.984±0.016 | 0.636±0.088 | 0/10 |
| fedce | 0.785±0.004 | 0.795±0.006 | 0.871±0.003 | 0.990±0.006 | 0.689±0.038 | 0/10 |
| proposed | 0.765±0.022 | 0.780±0.014 | 0.867±0.003 | 0.976±0.021 | 0.608±0.094 | 0/10 |

### 20% noisy clients (60% label flip)

| Method | Balanced acc | Accuracy | AUC | Jain | Worst-client | Collapsed seeds |
|---|---|---|---|---|---|---|
| fedavg | 0.720±0.093 | 0.713±0.122 | 0.860±0.027 | 0.899±0.153 | 0.510±0.251 | 1/10 |
| fedprox | 0.720±0.092 | 0.713±0.122 | 0.860±0.026 | 0.899±0.152 | 0.510±0.251 | 1/10 |
| feddyn | 0.761±0.032 | 0.763±0.035 | 0.863±0.008 | 0.975±0.021 | 0.592±0.087 | 0/10 |
| fedce | 0.749±0.085 | 0.739±0.112 | 0.859±0.031 | 0.935±0.147 | 0.583±0.209 | 1/10 |
| proposedcssv | 0.746±0.058 | 0.758±0.038 | 0.858±0.017 | 0.964±0.047 | 0.593±0.157 | 0/10 |
| proposed | 0.757±0.028 | 0.759±0.029 | 0.859±0.018 | 0.970±0.028 | 0.602±0.106 | 0/10 |

### 40% noisy clients (60% label flip)

| Method | Balanced acc | Accuracy | AUC | Jain | Worst-client | Collapsed seeds |
|---|---|---|---|---|---|---|
| fedavg | 0.649±0.116 | 0.663±0.112 | 0.846±0.042 | 0.816±0.169 | 0.333±0.282 | 3/10 |
| fedprox | 0.649±0.116 | 0.663±0.112 | 0.847±0.042 | 0.816±0.169 | 0.333±0.281 | 3/10 |
| feddyn | 0.697±0.107 | 0.717±0.084 | 0.850±0.020 | 0.908±0.121 | 0.461±0.262 | 2/10 |
| fedce | 0.657±0.129 | 0.671±0.120 | 0.845±0.047 | 0.815±0.203 | 0.369±0.312 | 4/10 |
| proposedcssv | 0.695±0.090 | 0.711±0.086 | 0.846±0.025 | 0.905±0.116 | 0.453±0.225 | 1/10 |
| proposed | 0.665±0.113 | 0.693±0.076 | 0.849±0.027 | 0.867±0.172 | 0.399±0.277 | 2/10 |

## Beyond accuracy: calibration, ranking quality, and threshold effects

`ece`/`brier` = calibration (lower better) — how trustworthy the predicted probability is, which is what matters for clinical triage. `auprc` = ranking quality on the positive (depressed) class. `bacc_opt` = balanced accuracy at the BEST threshold; the gap `bacc_opt − balanced_acc` is the loss due purely to a mis-placed operating point rather than to weaker discrimination.


### Clean

| Method | ECE ↓ | Brier ↓ | AUPRC ↑ | F1(pos) ↑ | bacc_opt ↑ | bacc_opt − balanced_acc |
|---|---|---|---|---|---|---|
| fedavg | 0.081±0.052 | 0.155±0.013 | 0.897±0.004 | 0.824±0.007 | 0.792±0.003 | +0.036 |
| fedprox | 0.081±0.052 | 0.155±0.013 | 0.897±0.004 | 0.824±0.007 | 0.792±0.003 | +0.036 |
| feddyn | 0.074±0.069 | 0.157±0.025 | 0.896±0.004 | 0.802±0.058 | 0.791±0.004 | +0.019 |
| fedce | 0.037±0.015 | 0.144±0.003 | 0.898±0.004 | 0.828±0.009 | 0.794±0.003 | +0.008 |
| proposed | 0.082±0.050 | 0.156±0.010 | 0.895±0.004 | 0.818±0.022 | 0.790±0.004 | +0.025 |

### 20% noisy clients

| Method | ECE ↓ | Brier ↓ | AUPRC ↑ | F1(pos) ↑ | bacc_opt ↑ | bacc_opt − balanced_acc |
|---|---|---|---|---|---|---|
| fedavg | 0.143±0.066 | 0.182±0.033 | 0.887±0.026 | 0.681±0.261 | 0.784±0.024 | +0.064 |
| fedprox | 0.143±0.066 | 0.182±0.033 | 0.887±0.025 | 0.682±0.261 | 0.784±0.023 | +0.064 |
| feddyn | 0.138±0.053 | 0.174±0.022 | 0.890±0.008 | 0.788±0.057 | 0.787±0.007 | +0.027 |
| fedce | 0.110±0.064 | 0.173±0.034 | 0.888±0.027 | 0.715±0.243 | 0.784±0.027 | +0.034 |
| proposedcssv | 0.127±0.037 | 0.172±0.019 | 0.884±0.022 | 0.796±0.027 | 0.785±0.012 | +0.039 |
| proposed | 0.141±0.032 | 0.175±0.017 | 0.886±0.019 | 0.783±0.048 | 0.784±0.014 | +0.027 |

### 40% noisy clients

| Method | ECE ↓ | Brier ↓ | AUPRC ↑ | F1(pos) ↑ | bacc_opt ↑ | bacc_opt − balanced_acc |
|---|---|---|---|---|---|---|
| fedavg | 0.181±0.065 | 0.206±0.036 | 0.876±0.037 | 0.672±0.236 | 0.772±0.036 | +0.123 |
| fedprox | 0.181±0.065 | 0.206±0.036 | 0.877±0.036 | 0.672±0.236 | 0.772±0.036 | +0.124 |
| feddyn | 0.155±0.065 | 0.190±0.031 | 0.878±0.019 | 0.764±0.092 | 0.780±0.015 | +0.083 |
| fedce | 0.152±0.068 | 0.198±0.039 | 0.875±0.042 | 0.682±0.234 | 0.772±0.041 | +0.115 |
| proposedcssv | 0.147±0.069 | 0.191±0.034 | 0.871±0.028 | 0.744±0.133 | 0.777±0.019 | +0.082 |
| proposed | 0.162±0.071 | 0.197±0.032 | 0.878±0.029 | 0.757±0.046 | 0.779±0.021 | +0.113 |

## The "opening": how much does noise cost each baseline?

| Method | Clean bal-acc | 40% bal-acc | Degradation |
|---|---|---|---|
| fedavg | 0.757 | 0.649 | +0.107 |
| fedprox | 0.757 | 0.649 | +0.108 |
| feddyn | 0.772 | 0.697 | +0.075 |
| fedce | 0.785 | 0.657 | +0.129 |
| proposed | 0.765 | 0.665 | +0.099 |

## Non-inferiority: proposed vs FedDyn (margin 0.01 balanced acc)

| Noise | Δ (proposed − FedDyn) | TOST p | Non-inferior at 0.01? |
|---|---|---|---|
| 0% | -0.0074 | 0.4040 | not established |
| 20% | -0.0034 | 0.3240 | not established |
| 40% | -0.0315 | 0.6579 | not established |

## Seed-matched head-to-head: proposed vs FedDyn

Same seed = same data partition and same corrupted-client set, so this pairs runs rather than comparing noisy means.

| Noise | Proposed wins | Mean Δ | Median Δ |
|---|---|---|---|
| 0% | 1/10 | -0.0074 | -0.0040 |
| 20% | 6/10 | -0.0034 | +0.0023 |
| 40% | 1/10 | -0.0315 | -0.0188 |

## Mechanism / ranking test (the pre-registered primary question)

Mean aggregation weight given to corrupted vs clean clients, pooled over all rounds and seeds. **Ratio < 1 means corrupted clients are correctly down-weighted.** FedAvg/FedDyn have no quality signal (size- or uniform-weighted), so they are the control.

| Method | Noise | Weight: corrupted | Weight: clean | Ratio | Probe-loss detection AUC |
|---|---|---|---|---|---|
| fedavg | 20% | 0.1813 | 0.1886 | **0.961** | — |
| fedavg | 40% | 0.1942 | 0.2000 | **0.971** | — |
| fedprox | 20% | 0.1813 | 0.1886 | **0.961** | — |
| fedprox | 40% | 0.1942 | 0.2000 | **0.971** | — |
| feddyn | 20% | 0.2000 | 0.2000 | **1.000** | — |
| feddyn | 40% | 0.2000 | 0.2000 | **1.000** | — |
| fedce | 20% | 0.1854 | 0.1877 | **0.988** | — |
| fedce | 40% | 0.2009 | 0.1945 | **1.033** | — |
| proposedcssv | 20% | 0.1926 | 0.1938 | **0.994** | — |
| proposedcssv | 40% | 0.1998 | 0.1973 | **1.013** | — |
| proposed | 20% | 0.1515 | 0.2047 | **0.740** | 0.917±0.063 |
| proposed | 40% | 0.1706 | 0.2178 | **0.783** | 0.874±0.088 |

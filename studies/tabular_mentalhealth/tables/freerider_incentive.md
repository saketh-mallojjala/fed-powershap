# C2/C3 — free-rider reward starvation and detection

10 seeds, clean labels (no label noise), free-rider fraction [0.1, 0.2, 0.3, 0.4]. Budget B=1 per round, split among that round's active clients by each rule, accumulated over 40 rounds.

**Pre-registered bars:** C2 — free-rider budget share ≤ 5% (equal-split ≈ 40%). C3 — detection AUC ≥ 0.85.


### Rung F1 — 10% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 5.4% ± 5.8 | 0.698 ± 0.135 | 0.833 ± 0.199 | fail | fail |
| blend_mult | 5.6% ± 6.3 | 0.744 ± 0.170 | 0.822 ± 0.206 | fail | fail |
| blend | 8.6% ± 5.8 | 0.732 ± 0.163 | 0.789 ± 0.270 | fail | fail |
| loss | 7.9% ± 5.8 | 0.679 ± 0.128 | 0.756 ± 0.171 | fail | fail |
| cssv | 10.7% ± 6.8 | 0.552 ± 0.073 | 0.544 ± 0.161 | fail | fail |
| effort | 7.0% ± 6.2 | 0.713 ± 0.182 | 0.756 ± 0.247 | fail | fail |
| norm | 8.5% ± 7.8 | 0.666 ± 0.160 | 0.656 ± 0.246 | fail | fail |
| size | 11.2% ± 11.0 | 0.608 ± 0.314 | 0.467 ± 0.306 | fail | fail |
| equal | 10.9% ± 7.2 | 0.500 ± 0.000 | 0.472 ± 0.278 | fail | fail |

### Rung F2 — 10% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 0.0% ± 0.0 | 0.826 ± 0.049 | 0.967 ± 0.037 | **PASS** | **PASS** |
| blend_mult | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.978 ± 0.044 | **PASS** | **PASS** |
| blend | 5.9% ± 3.8 | 0.936 ± 0.046 | 0.967 ± 0.071 | fail | **PASS** |
| loss | 7.9% ± 5.8 | 0.680 ± 0.129 | 0.756 ± 0.171 | fail | fail |
| cssv | 9.6% ± 6.2 | 0.596 ± 0.061 | 0.656 ± 0.153 | fail | fail |
| effort | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.978 ± 0.044 | **PASS** | **PASS** |
| norm | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.978 ± 0.044 | **PASS** | **PASS** |
| size | 11.2% ± 11.0 | 0.608 ± 0.314 | 0.467 ± 0.306 | fail | fail |
| equal | 10.9% ± 7.2 | 0.500 ± 0.000 | 0.472 ± 0.278 | fail | fail |

### Rung F3 — 10% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 2.1% ± 3.3 | 0.774 ± 0.082 | 0.900 ± 0.142 | **PASS** | **PASS** |
| blend_mult | 2.5% ± 3.2 | 0.832 ± 0.133 | 0.889 ± 0.157 | **PASS** | **PASS** |
| blend | 8.0% ± 5.5 | 0.769 ± 0.087 | 0.878 ± 0.189 | fail | **PASS** |
| loss | 8.6% ± 6.4 | 0.623 ± 0.161 | 0.667 ± 0.211 | fail | fail |
| cssv | 11.5% ± 8.5 | 0.492 ± 0.128 | 0.478 ± 0.254 | fail | fail |
| effort | 3.9% ± 3.6 | 0.807 ± 0.131 | 0.878 ± 0.168 | **PASS** | **PASS** |
| norm | 8.2% ± 10.8 | 0.749 ± 0.277 | 0.689 ± 0.306 | fail | fail |
| size | 11.2% ± 11.0 | 0.608 ± 0.314 | 0.467 ± 0.306 | fail | fail |
| equal | 10.9% ± 7.2 | 0.500 ± 0.000 | 0.472 ± 0.278 | fail | fail |

### Rung F4 — 10% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 12.1% ± 13.1 | 0.574 ± 0.129 | 0.628 ± 0.288 | fail | fail |
| blend_mult | 12.5% ± 13.3 | 0.531 ± 0.165 | 0.622 ± 0.295 | fail | fail |
| blend | 11.4% ± 8.0 | 0.512 ± 0.167 | 0.522 ± 0.337 | fail | fail |
| loss | 12.8% ± 10.4 | 0.491 ± 0.191 | 0.511 ± 0.295 | fail | fail |
| cssv | 12.4% ± 8.2 | 0.449 ± 0.119 | 0.444 ± 0.358 | fail | fail |
| effort | 9.0% ± 7.3 | 0.581 ± 0.115 | 0.644 ± 0.204 | fail | fail |
| norm | 8.9% ± 7.1 | 0.644 ± 0.165 | 0.600 ± 0.239 | fail | fail |
| size | 11.2% ± 11.0 | 0.608 ± 0.314 | 0.467 ± 0.306 | fail | fail |
| equal | 10.9% ± 7.2 | 0.500 ± 0.000 | 0.472 ± 0.278 | fail | fail |

### Rung F1 — 20% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 10.3% ± 7.9 | 0.665 ± 0.109 | 0.722 ± 0.170 | fail | fail |
| blend_mult | 10.2% ± 8.0 | 0.709 ± 0.143 | 0.719 ± 0.164 | fail | fail |
| blend | 16.8% ± 8.5 | 0.714 ± 0.131 | 0.775 ± 0.163 | fail | fail |
| loss | 17.2% ± 11.1 | 0.580 ± 0.181 | 0.575 ± 0.158 | fail | fail |
| cssv | 19.9% ± 9.4 | 0.585 ± 0.068 | 0.619 ± 0.178 | fail | fail |
| effort | 13.2% ± 7.6 | 0.696 ± 0.131 | 0.688 ± 0.165 | fail | fail |
| norm | 16.4% ± 10.6 | 0.661 ± 0.160 | 0.594 ± 0.188 | fail | fail |
| size | 21.6% ± 15.5 | 0.498 ± 0.303 | 0.456 ± 0.289 | fail | fail |
| equal | 20.9% ± 9.8 | 0.500 ± 0.000 | 0.394 ± 0.188 | fail | fail |

### Rung F2 — 20% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 0.0% ± 0.0 | 0.840 ± 0.032 | 0.956 ± 0.049 | **PASS** | **PASS** |
| blend_mult | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.963 ± 0.057 | **PASS** | **PASS** |
| blend | 11.8% ± 6.0 | 0.909 ± 0.066 | 0.925 ± 0.115 | fail | **PASS** |
| loss | 17.3% ± 11.1 | 0.580 ± 0.181 | 0.581 ± 0.163 | fail | fail |
| cssv | 18.3% ± 8.6 | 0.607 ± 0.074 | 0.706 ± 0.202 | fail | fail |
| effort | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.963 ± 0.057 | **PASS** | **PASS** |
| norm | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.963 ± 0.057 | **PASS** | **PASS** |
| size | 21.6% ± 15.5 | 0.498 ± 0.303 | 0.456 ± 0.289 | fail | fail |
| equal | 20.9% ± 9.8 | 0.500 ± 0.000 | 0.394 ± 0.188 | fail | fail |

### Rung F3 — 20% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 7.5% ± 6.7 | 0.712 ± 0.084 | 0.775 ± 0.229 | fail | fail |
| blend_mult | 6.6% ± 5.2 | 0.774 ± 0.092 | 0.794 ± 0.190 | fail | fail |
| blend | 16.9% ± 8.4 | 0.714 ± 0.145 | 0.775 ± 0.236 | fail | fail |
| loss | 18.1% ± 11.3 | 0.559 ± 0.185 | 0.537 ± 0.168 | fail | fail |
| cssv | 20.8% ± 11.6 | 0.530 ± 0.132 | 0.637 ± 0.227 | fail | fail |
| effort | 11.7% ± 10.0 | 0.720 ± 0.177 | 0.750 ± 0.255 | fail | fail |
| norm | 12.6% ± 9.7 | 0.757 ± 0.206 | 0.750 ± 0.196 | fail | fail |
| size | 21.6% ± 15.5 | 0.498 ± 0.303 | 0.456 ± 0.289 | fail | fail |
| equal | 20.9% ± 9.8 | 0.500 ± 0.000 | 0.394 ± 0.188 | fail | fail |

### Rung F4 — 20% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 26.2% ± 15.4 | 0.508 ± 0.152 | 0.447 ± 0.244 | fail | fail |
| blend_mult | 26.9% ± 15.4 | 0.470 ± 0.165 | 0.431 ± 0.200 | fail | fail |
| blend | 21.4% ± 10.6 | 0.506 ± 0.140 | 0.487 ± 0.238 | fail | fail |
| loss | 27.7% ± 12.8 | 0.328 ± 0.158 | 0.312 ± 0.265 | fail | fail |
| cssv | 19.7% ± 12.3 | 0.548 ± 0.116 | 0.531 ± 0.236 | fail | fail |
| effort | 16.8% ± 9.5 | 0.616 ± 0.102 | 0.669 ± 0.163 | fail | fail |
| norm | 17.9% ± 9.4 | 0.606 ± 0.141 | 0.600 ± 0.200 | fail | fail |
| size | 21.6% ± 15.5 | 0.498 ± 0.303 | 0.456 ± 0.289 | fail | fail |
| equal | 20.9% ± 9.8 | 0.500 ± 0.000 | 0.394 ± 0.188 | fail | fail |

### Rung F1 — 30% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 18.0% ± 10.6 | 0.645 ± 0.090 | 0.676 ± 0.127 | fail | fail |
| blend_mult | 16.1% ± 8.5 | 0.679 ± 0.113 | 0.681 ± 0.085 | fail | fail |
| blend | 25.4% ± 8.4 | 0.654 ± 0.091 | 0.638 ± 0.128 | fail | fail |
| loss | 22.7% ± 9.6 | 0.654 ± 0.170 | 0.600 ± 0.158 | fail | fail |
| cssv | 28.7% ± 9.7 | 0.582 ± 0.058 | 0.590 ± 0.146 | fail | fail |
| effort | 25.0% ± 12.9 | 0.598 ± 0.143 | 0.576 ± 0.150 | fail | fail |
| norm | 26.8% ± 16.5 | 0.571 ± 0.192 | 0.510 ± 0.257 | fail | fail |
| size | 27.2% ± 17.2 | 0.549 ± 0.230 | 0.495 ± 0.192 | fail | fail |
| equal | 29.4% ± 9.3 | 0.500 ± 0.000 | 0.471 ± 0.146 | fail | fail |

### Rung F2 — 30% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 0.0% ± 0.0 | 0.850 ± 0.033 | 0.950 ± 0.064 | **PASS** | **PASS** |
| blend_mult | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.957 ± 0.065 | **PASS** | **PASS** |
| blend | 16.1% ± 5.2 | 0.940 ± 0.046 | 0.943 ± 0.073 | fail | **PASS** |
| loss | 22.7% ± 9.6 | 0.654 ± 0.170 | 0.595 ± 0.160 | fail | fail |
| cssv | 25.5% ± 8.1 | 0.640 ± 0.082 | 0.690 ± 0.161 | fail | fail |
| effort | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.957 ± 0.065 | **PASS** | **PASS** |
| norm | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.957 ± 0.065 | **PASS** | **PASS** |
| size | 27.2% ± 17.2 | 0.549 ± 0.230 | 0.495 ± 0.192 | fail | fail |
| equal | 29.4% ± 9.3 | 0.500 ± 0.000 | 0.471 ± 0.146 | fail | fail |

### Rung F3 — 30% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 12.1% ± 10.9 | 0.698 ± 0.107 | 0.724 ± 0.143 | fail | fail |
| blend_mult | 10.8% ± 9.6 | 0.786 ± 0.129 | 0.733 ± 0.130 | fail | fail |
| blend | 24.9% ± 9.9 | 0.690 ± 0.104 | 0.667 ± 0.148 | fail | fail |
| loss | 23.3% ± 9.3 | 0.653 ± 0.172 | 0.595 ± 0.167 | fail | fail |
| cssv | 31.7% ± 12.6 | 0.465 ± 0.112 | 0.543 ± 0.198 | fail | fail |
| effort | 19.8% ± 15.3 | 0.711 ± 0.195 | 0.629 ± 0.212 | fail | fail |
| norm | 17.8% ± 13.5 | 0.770 ± 0.151 | 0.738 ± 0.158 | fail | fail |
| size | 27.2% ± 17.2 | 0.549 ± 0.230 | 0.495 ± 0.192 | fail | fail |
| equal | 29.4% ± 9.3 | 0.500 ± 0.000 | 0.471 ± 0.146 | fail | fail |

### Rung F4 — 30% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 29.0% ± 16.4 | 0.552 ± 0.161 | 0.557 ± 0.264 | fail | fail |
| blend_mult | 30.8% ± 19.0 | 0.539 ± 0.216 | 0.533 ± 0.282 | fail | fail |
| blend | 29.7% ± 10.8 | 0.510 ± 0.149 | 0.586 ± 0.221 | fail | fail |
| loss | 34.9% ± 17.0 | 0.422 ± 0.190 | 0.448 ± 0.224 | fail | fail |
| cssv | 31.6% ± 10.2 | 0.440 ± 0.130 | 0.467 ± 0.170 | fail | fail |
| effort | 22.6% ± 10.5 | 0.646 ± 0.146 | 0.700 ± 0.217 | fail | fail |
| norm | 27.5% ± 11.3 | 0.545 ± 0.094 | 0.486 ± 0.171 | fail | fail |
| size | 27.2% ± 17.2 | 0.549 ± 0.230 | 0.495 ± 0.192 | fail | fail |
| equal | 29.4% ± 9.3 | 0.500 ± 0.000 | 0.471 ± 0.146 | fail | fail |

### Rung F1 — 40% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 26.0% ± 11.1 | 0.613 ± 0.105 | 0.665 ± 0.130 | fail | fail |
| blend_mult | 25.8% ± 11.9 | 0.630 ± 0.119 | 0.687 ± 0.141 | fail | fail |
| blend | 34.9% ± 9.0 | 0.651 ± 0.123 | 0.650 ± 0.150 | fail | fail |
| loss | 27.5% ± 10.6 | 0.699 ± 0.182 | 0.637 ± 0.189 | fail | fail |
| cssv | 38.9% ± 8.4 | 0.568 ± 0.050 | 0.554 ± 0.205 | fail | fail |
| effort | 38.2% ± 14.4 | 0.519 ± 0.134 | 0.546 ± 0.207 | fail | fail |
| norm | 49.5% ± 20.1 | 0.322 ± 0.183 | 0.242 ± 0.223 | fail | fail |
| size | 36.2% ± 14.7 | 0.513 ± 0.212 | 0.492 ± 0.179 | fail | fail |
| equal | 39.3% ± 8.5 | 0.500 ± 0.000 | 0.496 ± 0.181 | fail | fail |

### Rung F2 — 40% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 0.0% ± 0.0 | 0.852 ± 0.039 | 0.983 ± 0.033 | **PASS** | **PASS** |
| blend_mult | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.967 ± 0.067 | **PASS** | **PASS** |
| blend | 20.9% ± 5.1 | 0.963 ± 0.030 | 0.958 ± 0.067 | fail | **PASS** |
| loss | 27.6% ± 10.7 | 0.699 ± 0.182 | 0.633 ± 0.182 | fail | fail |
| cssv | 35.1% ± 7.7 | 0.625 ± 0.064 | 0.696 ± 0.146 | fail | fail |
| effort | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.967 ± 0.067 | **PASS** | **PASS** |
| norm | 0.0% ± 0.0 | 1.000 ± 0.000 | 0.967 ± 0.067 | **PASS** | **PASS** |
| size | 36.2% ± 14.7 | 0.513 ± 0.212 | 0.492 ± 0.179 | fail | fail |
| equal | 39.3% ± 8.5 | 0.500 ± 0.000 | 0.496 ± 0.181 | fail | fail |

### Rung F3 — 40% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 26.9% ± 13.6 | 0.610 ± 0.107 | 0.667 ± 0.185 | fail | fail |
| blend_mult | 25.1% ± 11.7 | 0.649 ± 0.114 | 0.671 ± 0.191 | fail | fail |
| blend | 35.6% ± 9.6 | 0.621 ± 0.121 | 0.658 ± 0.222 | fail | fail |
| loss | 31.0% ± 9.6 | 0.670 ± 0.195 | 0.637 ± 0.188 | fail | fail |
| cssv | 38.5% ± 10.9 | 0.523 ± 0.075 | 0.579 ± 0.214 | fail | fail |
| effort | 37.3% ± 15.1 | 0.540 ± 0.162 | 0.537 ± 0.260 | fail | fail |
| norm | 40.9% ± 19.4 | 0.533 ± 0.200 | 0.588 ± 0.198 | fail | fail |
| size | 36.2% ± 14.7 | 0.513 ± 0.212 | 0.492 ± 0.179 | fail | fail |
| equal | 39.3% ± 8.5 | 0.500 ± 0.000 | 0.496 ± 0.181 | fail | fail |

### Rung F4 — 40% free-riders

| Reward signal | Free-rider budget share | Detection AUC (per round) | Detection AUC (cumulative) | C2 | C3 |
|---|---|---|---|---|---|
| gated | 44.2% ± 11.9 | 0.493 ± 0.097 | 0.529 ± 0.230 | fail | fail |
| blend_mult | 47.5% ± 13.3 | 0.441 ± 0.177 | 0.496 ± 0.208 | fail | fail |
| blend | 40.1% ± 8.6 | 0.471 ± 0.135 | 0.504 ± 0.226 | fail | fail |
| loss | 47.1% ± 10.1 | 0.346 ± 0.149 | 0.396 ± 0.077 | fail | fail |
| cssv | 39.9% ± 12.2 | 0.485 ± 0.103 | 0.417 ± 0.222 | fail | fail |
| effort | 33.2% ± 9.4 | 0.587 ± 0.097 | 0.679 ± 0.181 | fail | fail |
| norm | 40.1% ± 10.9 | 0.484 ± 0.102 | 0.467 ± 0.178 | fail | fail |
| size | 36.2% ± 14.7 | 0.513 ± 0.212 | 0.492 ± 0.179 | fail | fail |
| equal | 39.3% ± 8.5 | 0.500 ± 0.000 | 0.496 ± 0.181 | fail | fail |

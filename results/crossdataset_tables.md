# Cross-dataset results (3 seeds, last round; APTOS also reports QWK)


### Clean (0% noise)

| Dataset | Method | Accuracy | Jain | Worst-client |
|---|---|---|---|---|
| OCTMNIST | FedAvg | 0.636±0.057 (n=5) | 0.949 | 0.427 |
| OCTMNIST | FedDyn | 0.688±0.035 (n=5) | 0.976 | 0.594 |
| OCTMNIST | Proposed | 0.708±0.029 (n=5) | 0.969 | 0.536 |
| ISIC | FedAvg | 0.697±0.001 (n=3) | 0.694 | 0.026 |
| ISIC | FedDyn | 0.734±0.013 (n=3) | 0.807 | 0.108 |
| ISIC | Proposed | 0.719±0.019 (n=3) | 0.782 | 0.101 |
| APTOS | FedAvg | 0.678±0.011 (n=3) | 0.754 | 0.048 |
| APTOS | FedDyn | 0.715±0.023 (n=3) | 0.842 | 0.180 |
| APTOS | Proposed | 0.728±0.014 (n=3) | 0.822 | 0.075 |

### 40% label noise

| Dataset | Method | Accuracy | Jain | Worst-client |
|---|---|---|---|---|
| OCTMNIST | FedAvg | 0.462±0.054 (n=4) | 0.885 | 0.189 |
| OCTMNIST | FedDyn | 0.523±0.075 (n=4) | 0.926 | 0.324 |
| OCTMNIST | Proposed | 0.518±0.042 (n=3) | 0.904 | 0.215 |
| ISIC | FedAvg | 0.669±0.007 (n=3) | 0.703 | 0.020 |
| ISIC | FedDyn | 0.672±0.026 (n=3) | 0.825 | 0.153 |
| ISIC | Proposed | 0.645±0.046 (n=3) | 0.809 | 0.130 |
| APTOS | FedAvg | 0.577±0.083 (n=3) | 0.716 | 0.000 |
| APTOS | FedDyn | 0.649±0.084 (n=3) | 0.816 | 0.152 |
| APTOS | Proposed | 0.588±0.133 (n=3) | 0.816 | 0.048 |

# ShapFed + Power-of-Choice

## APTOS 2019 quickstart

```bash
# 1. Kaggle credentials at ~/.kaggle/kaggle.json (one-time)
#    https://www.kaggle.com/settings → "Create New API Token"
# 2. Accept competition rules in browser (one-time):
#    https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules

pip install -r requirements.txt
python scripts/download_aptos.py        # ~10 GB, into data_cache/aptos2019/
python scripts/aptos_smoke.py           # 2-round sanity check

# Full APTOS run (20 clients, ResNet-18 pretrained, 100 rounds)
python main.py \
  --dataset aptos --model resnet18 --num-classes 5 \
  --image-size 224 --batch-size 32 --num-workers 4 \
  --num-clients 20 --alpha 0.3 --num-rounds 100 \
  --candidate-size-d 10 --active-size-m 5 --local-epochs 2 --local-lr 0.001
```

`evaluate()` reports **Quadratic Weighted Kappa (QWK)** alongside accuracy
when sklearn is installed — that's APTOS' official metric since DR grades
0–4 are ordinal.

## Overview

Federated learning that combines two complementary pieces of prior work:

1. **Power-of-Choice** (Cho et al., 2020) for *client selection*: each round,
   sample a candidate pool of size `d`, ask each candidate for its local loss
   on the current global model, then train only the top-`m` highest-loss
   clients. This biases training toward the clients the current model is
   doing worst on.
2. **ShapFed** (Tastan et al., IJCAI 2024) for *aggregation*: compute
   class-specific Shapley-like contributions (CSSV) from each client's
   last-layer update, and use them to weight the aggregation (ShapFed-WA).

The original repos are [peng-ju/Power-of-Choice](https://github.com/peng-ju/Power-of-Choice)
and [tnurbek/shapfed](https://github.com/tnurbek/shapfed).

## Algorithm (per round `t`)

```
1. Candidate pool:  A_t ~ sample d clients, p_k ∝ |D_k|          (Power-of-Choice)
2. Probe losses:    L_k = F_k(w_t)  for k ∈ A_t                  (Power-of-Choice)
3. Active set:      S_t = top-m clients in A_t by L_k            (Power-of-Choice)
4. Local training:  w_k ← SGD(w_t, D_k, E epochs)                (vanilla)
5. CSSV:            φ_{k,c} = cos(Δw_k^{(c)}, Δw_agg^{(c)})       (ShapFed)
6. Weights:         w_k ∝ (Σ_c φ_{k,c})_{>0} · (|D_k| / Σ|D_j|)   (ShapFed-WA + size prior)
7. Aggregate:       w_{t+1} = Σ_k w_k · w_k^local                (FedAvg-style)
```

`Δw_k^{(c)}` is row `c` of the final classifier layer's weight update —
exactly the "class-c gradient" ShapFed uses.

## Layout

```
shapfed-poc/
├── config.py              # Dataclass of all hyperparameters + CLI-friendly
├── main.py                # Entry point
├── data/dataset.py        # CIFAR-10/MNIST loaders + Dirichlet partition
├── models/cnn.py          # Small CNN (classifier layer name is fixed)
├── federated/
│   ├── selection.py       # Power-of-Choice (+ random/full baselines)
│   ├── shapley.py         # CSSV + reduction to aggregation weights
│   ├── aggregation.py     # Weighted state_dict averaging
│   ├── client.py          # Local SGD, loss probe, class histogram
│   └── server.py          # Round loop, evaluation
├── utils/logger.py        # JSONL logger, seed helper
├── scripts/smoke_test.py  # 2-round MNIST sanity check
└── logs/                  # JSONL logs written here
```

## Quick start

```bash
pip install -r requirements.txt

# Sanity check (small, CPU, ~1 min)
python scripts/smoke_test.py

# Full run (CIFAR-10, 20 clients, non-IID Dirichlet α=0.3)
python main.py

# Ablations
python main.py --selection-strategy random              # no pow-d
python main.py --aggregation fedavg                     # no CSSV
python main.py --selection-strategy random --aggregation fedavg  # vanilla FedAvg
```

All flags mirror `config.py`. Use `--no-size-weighted-candidates`, etc. for
boolean flags (argparse BooleanOptionalAction).

## Output

Each run produces a JSONL log under `logs/`. One record per line:

- `{"type": "config", ...}` — full resolved config
- `{"type": "client_sizes", ...}` — per-client sample counts + class histograms
- `{"type": "round", "round": r, "candidates": [...], "candidate_losses": [...], "active": [...], "weights": [...], "cssv": [[...]]}` — selection + aggregation detail
- `{"type": "eval", "round": r, "acc": ..., "per_class_acc": [...]}` — evaluation metrics

That's enough to reproduce plots of accuracy vs. round, CSSV dynamics, and
the Power-of-Choice distribution over time.

## Knobs that matter

| Flag | Meaning | Typical |
|---|---|---|
| `--alpha` | Dirichlet concentration (lower = more non-IID) | 0.1 – 1.0 |
| `--candidate-size-d` | Power-of-Choice candidate pool size | 2× `m` to `N` |
| `--active-size-m` | Clients trained per round | 5 – 10 for N=20 |
| `--cssv-temperature` | Softmax sharpness on CSSV scores | 1.0 (linear norm) |
| `--cssv-ema` | EMA smoothing on per-client CSSV | 0.3 – 0.7 |
| `--no-cssv-clamp-negative` | Allow negative CSSV (gradients opposing aggregate) | off |

## Design notes

- CSSV uses the last classifier layer's weight delta because it decomposes
  cleanly per class — row `c` is the "class-`c` contribution". That matches
  ShapFed's gradient-projection formulation without needing to materialize
  per-sample gradients.
- The CSSV → weight reduction blends the Shapley score with a size prior
  (FedAvg weighting). Pure CSSV can be unstable early in training when all
  deltas point roughly the same direction and cosine similarities cluster.
- Power-of-Choice's `probe_loss` uses only 4 mini-batches per candidate per
  round; that's cheap enough to run on every candidate without dominating
  wall time.
- EMA on CSSV (`--cssv-ema`) stabilizes aggregation weights across rounds so
  a client isn't "punished" for one noisy round.

## Extending

- **Different datasets**: add a branch in `data/dataset.py::_load_raw`.
- **Different model**: add to `models/` and make sure the last `nn.Linear` is
  named `classifier` (this constant lives in `models/__init__.py`).
- **Different selection rule** (e.g. FedCor, Oort): add a branch in
  `federated/selection.py::select_clients`.
- **Different CSSV reduction** (e.g. per-class reweighting by class rarity):
  extend `federated/shapley.py::cssv_to_weights`.

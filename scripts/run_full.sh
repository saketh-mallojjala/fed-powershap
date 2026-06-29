#!/usr/bin/env bash
# Full sweep over the cached MedMNIST datasets, then build tables + figures.
# APTOS is omitted until Kaggle data is downloaded (add 'aptos' to DATASETS then).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$(command -v python3)}"

DATASETS="${DATASETS:-octmnist isic chestxray14}"
SEEDS="${SEEDS:-0 1 2}"
# Config: 10 clients with healthy per-client data (20k cap) — the regime where
# contribution/selection signals are reliable; 2 local epochs, batch 128.
ROUNDS="${ROUNDS:-40}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
MAX_TRAIN="${MAX_TRAIN:-20000}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
EVAL_EVERY="${EVAL_EVERY:-2}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
BATCH="${BATCH:-128}"
NOISY_FRAC="${NOISY_FRAC:-0.4}"
NOISE_RATE="${NOISE_RATE:-0.6}"

echo "[run_full] datasets=$DATASETS seeds=$SEEDS rounds=$ROUNDS img=$IMAGE_SIZE eval_every=$EVAL_EVERY le=$LOCAL_EPOCHS bs=$BATCH noisy=$NOISY_FRAC@$NOISE_RATE"

"$PY" scripts/run_experiments.py \
  --datasets $DATASETS --seeds $SEEDS \
  --rounds "$ROUNDS" --image-size "$IMAGE_SIZE" \
  --max-train "$MAX_TRAIN" --num-clients "$NUM_CLIENTS" \
  --eval-every "$EVAL_EVERY" --local-epochs "$LOCAL_EPOCHS" --batch-size "$BATCH" \
  --noisy-client-frac "$NOISY_FRAC" --label-noise-rate "$NOISE_RATE"

echo "[run_full] building results table..."
"$PY" scripts/aggregate_results.py --datasets $DATASETS

echo "[run_full] building per-dataset comparison figures..."
for ds in $DATASETS; do
  logs=$(ls logs/${ds}_*_seed0-*.jsonl 2>/dev/null)
  if [ -n "$logs" ]; then
    "$PY" scripts/plot_results.py $logs || true
  fi
done

echo "[run_full] DONE. Table: results/results_table.md  Figures: figures/"

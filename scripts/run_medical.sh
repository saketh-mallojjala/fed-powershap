#!/usr/bin/env bash
# Run both experiment arms (ShapFed+pow-d vs FedAvg+random) for each medical
# dataset, then generate the full figure set per dataset — same protocol we
# used for APTOS.
#
# Usage:
#   bash scripts/run_medical.sh                       # all MedMNIST datasets
#   bash scripts/run_medical.sh octmnist isic         # a subset
#   ROUNDS=50 IMAGE_SIZE=128 bash scripts/run_medical.sh octmnist
#
# BraTS requires the slice cache first:
#   python scripts/prep_brats_slices.py --src /path/to/BraTS/training
#   bash scripts/run_medical.sh brats
set -euo pipefail

cd "$(dirname "$0")/.."

# Use python3 if no `python` on PATH (common on macOS).
PY="${PYTHON:-$(command -v python || command -v python3)}"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(octmnist isic chestxray14)
fi

ROUNDS="${ROUNDS:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
ALPHA="${ALPHA:-0.3}"
LR="${LR:-0.001}"
MAX_TRAIN="${MAX_TRAIN:-0}"        # stratified cap on train pool; 0 = use all
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"    # 0 best for in-memory MedMNIST (no worker hang)

common=(--model resnet18 --image-size "$IMAGE_SIZE" \
        --num-clients "$NUM_CLIENTS" --alpha "$ALPHA" --num-rounds "$ROUNDS" \
        --candidate-size-d 10 --active-size-m 5 --local-epochs "$LOCAL_EPOCHS" \
        --local-lr "$LR" --max-train-samples "$MAX_TRAIN" --num-workers "$NUM_WORKERS")

for ds in "${DATASETS[@]}"; do
  echo "================  $ds  ================"

  # Arm 1: ShapFed CSSV weighting + Power-of-Choice selection.
  "$PY" main.py --dataset "$ds" "${common[@]}" \
    --aggregation shapfed_wa --selection-strategy pow_d \
    --exp-name "${ds}_shapfed_pow"

  # Arm 2: FedAvg + random selection baseline.
  "$PY" main.py --dataset "$ds" "${common[@]}" \
    --aggregation fedavg --selection-strategy random \
    --exp-name "${ds}_fedavg_random"

  # Figures comparing the two arms.
  shap_log=$(ls -t logs/${ds}_shapfed_pow-*.jsonl | head -1)
  base_log=$(ls -t logs/${ds}_fedavg_random-*.jsonl | head -1)
  "$PY" scripts/plot_results.py "$shap_log" "$base_log"
done

echo "All done. Figures are under figures/."

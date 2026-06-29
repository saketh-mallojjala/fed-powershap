#!/usr/bin/env bash
# Wait for the MedMNIST sweep to finish, then run the APTOS sweep at 128px
# (fundus detail matters for QWK), then rebuild the full 4-dataset table + figures.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$(command -v python3)}"

echo "[aptos_after] waiting for MedMNIST sweep (run_experiments.py) to finish..."
while pgrep -f "run_experiments.py" >/dev/null 2>&1; do sleep 60; done
echo "[aptos_after] MedMNIST sweep done; starting APTOS at 128px"

"$PY" scripts/run_experiments.py \
  --datasets aptos --seeds 0 1 2 \
  --rounds 40 --image-size 128 --eval-every 2 \
  --local-epochs 2 --batch-size 64 --num-clients 10 --num-workers 2

echo "[aptos_after] rebuilding full table (all 4 datasets)..."
"$PY" scripts/aggregate_results.py

for ds in octmnist isic chestxray14 aptos; do
  logs=$(ls logs/${ds}_*_seed0-*.jsonl 2>/dev/null)
  [ -n "$logs" ] && "$PY" scripts/plot_results.py $logs || true
done

echo "[aptos_after] ALL DONE. Table: results/results_table.md  Figures: figures/"

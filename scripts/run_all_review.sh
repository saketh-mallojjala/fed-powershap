#!/usr/bin/env bash
# Full revision experiment queue, in reviewer-priority order.
# Resumable: run_review_sweep.py skips tags already recorded, so re-running
# this script after an interruption continues where it stopped.
set -u
cd "$(dirname "$0")/.."
SW="python3 scripts/run_review_sweep.py --device mps"

# Wait for any sweep already in flight so the queue never double-runs.
while pgrep -f "run_review_sweep.py" > /dev/null; do sleep 30; done

echo "=== APTOS tier 1 (corruption level x method) ==="
$SW --dataset aptos --tier 1 --seeds 0 1
echo "=== APTOS tier 2 (A.2 mixed, noise types, poisoning) ==="
$SW --dataset aptos --tier 2 --seeds 0
echo "=== APTOS tier 3 (alpha, participation, scaling) ==="
$SW --dataset aptos --tier 3 --seeds 0
echo "=== ISIC tier 1 ==="
$SW --dataset isic  --tier 1 --seeds 0
echo "=== ISIC tier 2 ==="
$SW --dataset isic  --tier 2 --seeds 0
echo "=== OCTMNIST tier 1 (20k cap) ==="
$SW --dataset octmnist --tier 1 --seeds 0
echo "=== OCTMNIST tier 2 (20k cap) ==="
$SW --dataset octmnist --tier 2 --seeds 0
# More seeds for the headline grid, appended LAST: the other reviewer points
# are unanswered until their runs exist, whereas this one is merely
# under-powered. Paired per-seed differences have sd ~0.007 against an
# unpaired arm sd of ~0.045, so extra seeds buy real confidence cheaply.
echo "=== APTOS tier 1, seeds 2-4 (power for the headline claim) ==="
$SW --dataset aptos --tier 1 --seeds 2 3 4
echo "ALL_REVIEW_DONE"

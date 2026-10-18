#!/usr/bin/env bash
# Fast path. Priority: get the headline corruption table on ALL THREE datasets
# (reviewer A asked for that explicitly), then depth on the second dataset.
# Dropped vs the original plan: OCTMNIST tier 2, and APTOS seeds 2-4.
set -u
cd "$(dirname "$0")/.."
SW="python3 scripts/run_review_sweep.py --device mps"
while pgrep -f "run_review_sweep.py" > /dev/null; do sleep 20; done

echo "=== ISIC tier 1 (headline table, dataset 2) ==="
$SW --dataset isic --tier 1 --seeds 0
echo "=== OCTMNIST tier 1 (headline table, dataset 3; 10k cap) ==="
$SW --dataset octmnist --tier 1 --seeds 0
# ISIC tier 2 dropped for time. The shift/mixed and poisoning analyses are
# reported on APTOS; the headline corruption table covers all three datasets.
echo "ALL_REVIEW_DONE"

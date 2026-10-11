#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE="$PROJECT_ROOT/data/processed/cold_start_exact_dataset.pt"
OUTPUT="$PROJECT_ROOT/outputs/journal_v1/cold_start_exact"
SELECTIVE_OUTPUT="$PROJECT_ROOT/outputs/journal_v1/selective_exact"

"$PYTHON_BIN" "$PROJECT_ROOT/experiments/train_cold_start_anytime.py" \
  --cache "$CACHE" --output-root "$OUTPUT" --prepare-only

for seed in 20260820 20260821 20260822 20260823 20260824; do
  "$PYTHON_BIN" "$PROJECT_ROOT/experiments/train_cold_start_anytime.py" \
    --cache "$CACHE" --output-root "$OUTPUT" --method hybrid --run-seed "$seed"
  "$PYTHON_BIN" "$PROJECT_ROOT/experiments/train_cold_start_anytime.py" \
    --cache "$CACHE" --output-root "$OUTPUT" --method temporal_only --run-seed "$seed"
done

for seed in 20260820 20260821 20260822 20260823 20260824; do
  "$PYTHON_BIN" "$PROJECT_ROOT/experiments/train_cold_start_anytime.py" \
    --cache "$CACHE" --output-root "$SELECTIVE_OUTPUT" \
    --method selective_hybrid --run-seed "$seed"
done

"$PYTHON_BIN" "$PROJECT_ROOT/experiments/evaluate_cold_start_anytime.py" \
  --checkpoint-root "$SELECTIVE_OUTPUT" \
  --temporal-checkpoint-root "$OUTPUT" \
  --hybrid-method selective_hybrid \
  --training-cache "$CACHE" \
  --test-cache "$PROJECT_ROOT/data/processed/cold_start_selective_replication.pt" \
  --test-seed-base 1510000 \
  --output-dir "$PROJECT_ROOT/outputs/journal_v1/selective_replication"

"$PYTHON_BIN" "$PROJECT_ROOT/experiments/train_reference_baseline.py" \
  --architecture tgm_tgn --seed 20260820 --epochs 20 --cache "$CACHE" \
  --out-dir "$PROJECT_ROOT/outputs/journal_v1/reference_baselines/tgm_tgn"

"$PYTHON_BIN" "$PROJECT_ROOT/experiments/evaluate_journal_robustness.py" \
  --checkpoint-root "$SELECTIVE_OUTPUT" \
  --temporal-checkpoint-root "$OUTPUT" \
  --hybrid-method selective_hybrid --training-cache "$CACHE" \
  --output-dir "$PROJECT_ROOT/outputs/journal_v1/robustness" \
  --replicates 2 --snapshot-ensemble-size 8 --seed-base 1810000

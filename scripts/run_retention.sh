#!/usr/bin/env bash
# Retention-mechanism grid: {replay off/on} x {soft ortho / hard projection},
# controller only. Artifacts land under artifacts/sweeps/retention/ so
# summarize_sweeps.py picks them up alongside the earlier sweeps.
#
# Override any list from the shell:
#   SEEDS="0 1 2" REPLAY_VALUES="0 0.3 1.0" bash scripts/run_retention.sh

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-artifacts/sweeps/retention}"
STEPS="${STEPS:-200}"
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
REPLAY_VALUES=(${REPLAY_VALUES:-0 1.0})

mkdir -p "${OUT_ROOT}"

echo "Writing retention-grid artifacts under ${OUT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Steps: ${STEPS}"
echo "Seeds: ${SEEDS[*]-}"
echo "Replay weights: ${REPLAY_VALUES[*]-}"

set +u
for seed in "${SEEDS[@]}"; do
  for replay in "${REPLAY_VALUES[@]}"; do
    rtag="${replay//./p}"
    # Whole grid runs --no-gates: gates were inert in earlier sweeps and
    # hard projection is only exact without them (single-variable arms).
    "${PYTHON_BIN}" scripts/run_controller.py \
      --steps "${STEPS}" --seed "${seed}" --replay "${replay}" --no-gates \
      --out "${OUT_ROOT}/controller_replay_${rtag}_seed_${seed}.json"
    "${PYTHON_BIN}" scripts/run_controller.py \
      --steps "${STEPS}" --seed "${seed}" --replay "${replay}" --no-gates \
      --hard-ortho \
      --out "${OUT_ROOT}/controller_replay_${rtag}_hard_seed_${seed}.json"
  done
done
set -u

echo
echo "Done. Aggregate with: python scripts/summarize_sweeps.py --stdout"

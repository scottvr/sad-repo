#!/usr/bin/env bash
# Run follow-up sweeps for low-dimensional composition experiments.
#
# Defaults are GPU-friendly but non-trivial. Override any list from the shell:
#   SEEDS="0 1 2" K_VALUES="4 8 16" bash scripts/run_sweeps.sh

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-artifacts/sweeps}"
STEPS="${STEPS:-200}"
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
K_VALUES=(${K_VALUES:-4 8 16 32})
ORTHO_VALUES=(${ORTHO_VALUES:-0 0.1 1.0})
TASK_COUNTS=(${TASK_COUNTS:-2 3 4 5 6})

mkdir -p "${OUT_ROOT}"

run_sequence() {
  local out="$1"
  shift
  "${PYTHON_BIN}" scripts/evaluate_sequence.py \
    --steps "${STEPS}" \
    "$@" \
    --out "${out}"
}

run_controller() {
  local out="$1"
  shift
  "${PYTHON_BIN}" scripts/run_controller.py \
    --steps "${STEPS}" \
    "$@" \
    --out "${out}"
}

echo "Writing sweep artifacts under ${OUT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Steps: ${STEPS}"
echo "Seeds: ${SEEDS[*]}"

echo
echo "== Seed sweep: full suite + anchored controller =="
mkdir -p "${OUT_ROOT}/seed"
for seed in "${SEEDS[@]}"; do
  run_sequence "${OUT_ROOT}/seed/sequence_seed_${seed}.json" \
    --seed "${seed}"
  run_controller "${OUT_ROOT}/seed/controller_anchor_seed_${seed}.json" \
    --seed "${seed}" --anchor 1.0
done

echo
echo "== K sweep: full suite over n_components =="
mkdir -p "${OUT_ROOT}/k"
for k in "${K_VALUES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_sequence "${OUT_ROOT}/k/sequence_k_${k}_seed_${seed}.json" \
      --seed "${seed}" --n-components "${k}"
  done
done

echo
echo "== Ortho sweep: controller only =="
mkdir -p "${OUT_ROOT}/ortho"
for ortho in "${ORTHO_VALUES[@]}"; do
  tag="${ortho//./p}"
  for seed in "${SEEDS[@]}"; do
    run_controller "${OUT_ROOT}/ortho/controller_ortho_${tag}_seed_${seed}.json" \
      --seed "${seed}" --ortho "${ortho}"
  done
done

echo
echo "== No-gates ablation: controller only =="
mkdir -p "${OUT_ROOT}/no_gates"
for seed in "${SEEDS[@]}"; do
  run_controller "${OUT_ROOT}/no_gates/controller_no_gates_seed_${seed}.json" \
    --seed "${seed}" --no-gates
  run_controller "${OUT_ROOT}/no_gates/controller_anchor_no_gates_seed_${seed}.json" \
    --seed "${seed}" --anchor 1.0 --no-gates
done

echo
echo "== Task-count sweep: full suite =="
mkdir -p "${OUT_ROOT}/task_count"
for n_tasks in "${TASK_COUNTS[@]}"; do
  facts_per_task=4
  if [[ "${n_tasks}" -gt 4 ]]; then
    facts_per_task=3
  fi
  for seed in "${SEEDS[@]}"; do
    run_sequence "${OUT_ROOT}/task_count/sequence_tasks_${n_tasks}_facts_${facts_per_task}_seed_${seed}.json" \
      --seed "${seed}" --n-tasks "${n_tasks}" --facts-per-task "${facts_per_task}"
  done
done

echo
echo "Done. Aggregate with scripts/summarize_sweeps.py or the narrower seed-only summarizer."

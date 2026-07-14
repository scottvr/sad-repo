#!/usr/bin/env bash
# Dims grid: is 96 a magic number, or just n_sites x k?
#
# Total coefficient dims = n_sites * k. The cap arms of the pressure grid
# already trace the both-sites curve (k=2/4/16 -> 24/48/192 dims; 96 is the
# retention grid's replay=1 arm). This grid varies the OTHER axis — which
# sites get the budget — so the same dim count is realized by different
# allocations. On distilgpt2 (6 layers), 48 dims appears five ways:
#
#   attn k=8   (attention only, all layers)      \
#   mlp  k=8   (MLP only, all layers)             | same budget,
#   both k=4   (pressure grid cap_k4)             | different allocation
#   early k=8  (both sites, layers 0-2)           |
#   late  k=8  (both sites, layers 3-5)          /
#
# If retention/fit track total dims regardless of allocation, dims is the
# resource and 96 is nothing special. If allocation matters (e.g. late
# layers or MLP sites dominate), the story is about WHERE capacity sits,
# and the size/complexity relationship must be sought per-site-type.
# All arms replay=1 --no-gates (the going-forward default).
#
# Override lists from the shell:
#   SEEDS="0 1 2" STEPS=200 MODEL=distilgpt2 bash scripts/run_dims.sh

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-artifacts/sweeps/dims}"
STEPS="${STEPS:-200}"
MODEL="${MODEL:-distilgpt2}"
SEEDS=(${SEEDS:-0 1 2})

mkdir -p "${OUT_ROOT}"

echo "Writing dims-grid artifacts under ${OUT_ROOT}"
echo "Python: ${PYTHON_BIN}  Model: ${MODEL}  Steps: ${STEPS}"
echo "Seeds: ${SEEDS[*]-}"

run_arm() {
  local outname=$1; shift
  local seed=$1; shift
  "${PYTHON_BIN}" scripts/run_controller.py \
    --model "${MODEL}" --steps "${STEPS}" --seed "${seed}" \
    --no-gates --replay 1.0 "$@" \
    --out "${OUT_ROOT}/controller_${outname}_seed_${seed}.json"
}

set +u
for seed in "${SEEDS[@]}"; do
  # dims curve on attention sites only: 12 / 24 / 48 / 96 dims
  for k in 2 4 8 16; do
    run_arm "attn_k${k}" "${seed}" --sites attn --k "${k}"
  done
  # allocation controls at 48 dims (see header)
  run_arm "mlp_k8"   "${seed}" --sites mlp
  run_arm "early_k8" "${seed}" --layers 0-2
  run_arm "late_k8"  "${seed}" --layers 3-5
done
set -u

echo
echo "Done. Aggregate with: python scripts/summarize_sweeps.py --stdout"

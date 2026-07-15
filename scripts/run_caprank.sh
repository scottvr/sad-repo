#!/usr/bin/env bash
# Capacity-vs-scale grid: does paraphrase transfer emerge with update
# capacity, at fixed model scale?
#
# Motivation (corrected coherence re-run, 2026-07-15, CURRENT_STATUS
# verdicts 18-19): coefficient updates (k=8, rank 4, 96 dims) show
# ~chance transfer to the held-out template, but the full-rank
# naive_stack LoRA baseline scored .88 paraphrase consistency in the
# same runs. If transfer is gated by capacity rather than model scale,
# raising k (more basis components) and/or rank (richer components)
# should lift the composed state's held-out-template ACCURACY — the
# `final_evals_heldout` field / `heldout acc` summary column (recorded
# since 2026-07-15; consistency alone can be inflated by agreeing-wrong
# states). If heldout acc stays at chance while k*rank grows toward
# full-rank territory, capacity is exonerated and the TinyLlama scale
# arc (roadmap v0.3 Arc 3) is back on.
#
# Two axes from the same anchor (k=8 r=4; both sites; replay=1; no-gates):
#   components: k = 8 / 16 / 32 / 64  at rank 4   (96 -> 768 dims)
#   rank:       r = 4 / 16            at k 8/16   (component richness)
#   combined:   k=32 r=16                          (approaching LoRA regime)
#
# The naive_stack LoRA reference (rank 4, full A/B trainable) comes from
# the multiseed suite (evaluate_sequence), not this grid — rerun that
# with post-2026-07-15 code to populate its heldout column.
#
# Override lists from the shell:
#   SEEDS="0 1 2 3 4" STEPS=200 MODEL=distilgpt2 bash scripts/run_caprank.sh

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-artifacts/sweeps/caprank}"
STEPS="${STEPS:-200}"
MODEL="${MODEL:-distilgpt2}"
SEEDS=(${SEEDS:-0 1 2 3 4})

mkdir -p "${OUT_ROOT}"

echo "Writing caprank-grid artifacts under ${OUT_ROOT}"
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
  # components axis (rank 4): 96 / 192 / 384 / 768 dims
  for k in 8 16 32 64; do
    run_arm "k${k}_r4" "${seed}" --k "${k}"
  done
  # rank axis at fixed k
  run_arm "k8_r16"  "${seed}" --rank 16
  run_arm "k16_r16" "${seed}" --k 16 --rank 16
  # combined: k*rank = 512 per site, well into the LoRA capacity regime
  run_arm "k32_r16" "${seed}" --k 32 --rank 16
done
set -u

echo
echo "Done. Aggregate with: python scripts/summarize_sweeps.py --stdout"
echo "Read the 'heldout acc' column: rising with k/rank => capacity gates"
echo "paraphrase transfer; flat at chance => scale (or the linear-update"
echo "form itself) is the constraint."

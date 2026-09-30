#!/usr/bin/env bash
# Final session: pre-registered stop/continue test (docs/final_session.md).
#
#   1. Prompting baseline (facts pasted in context, no adapters) on
#      distilgpt2 / gpt2 / gpt2-medium x default / conflict / big family.
#      Inference only: minutes.
#   2. Coefficient adapters, distilgpt2, replay=1: diversity axis
#      (2/4/8 training phrasings) x capacity axis (k, rank).
#   3. LoRA (naive_stack) reference at 2 and 8 training phrasings.
#
# Then: python scripts/summarize_final.py  (tables + mechanical verdict)
#
# Override from the shell:
#   SEEDS="0 1" bash scripts/run_final.sh             # pilot
#   PARTS="icl" bash scripts/run_final.sh             # prompting only
#   ICL_MODELS="distilgpt2 gpt2" bash scripts/run_final.sh

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-artifacts/final}"
STEPS="${STEPS:-200}"
SEEDS=(${SEEDS:-0 1 2 3 4})
ICL_MODELS=(${ICL_MODELS:-distilgpt2 gpt2 gpt2-medium})
PARTS="${PARTS:-icl coef lora}"

mkdir -p "${OUT_ROOT}"
echo "Writing final-session artifacts under ${OUT_ROOT}"
echo "Python: ${PYTHON_BIN}  Steps: ${STEPS}  Seeds: ${SEEDS[*]-}  Parts: ${PARTS}"

has_part() { [[ " ${PARTS} " == *" $1 "* ]]; }

# Skip runs whose artifact already exists, so a Colab disconnect can resume.
run() {
  local out=$1; shift
  if [[ -f "${out}" ]]; then echo "skip (exists): ${out}"; return; fi
  "${PYTHON_BIN}" "$@" --out "${out}"
}

set +u
if has_part icl; then
  for model in "${ICL_MODELS[@]}"; do
    tag="${model//\//_}"
    for seed in "${SEEDS[@]}"; do
      run "${OUT_ROOT}/icl_${tag}_default_seed_${seed}.json" \
        scripts/run_icl_baseline.py --model "${model}" --seed "${seed}"
      run "${OUT_ROOT}/icl_${tag}_conflict_seed_${seed}.json" \
        scripts/run_icl_baseline.py --model "${model}" --seed "${seed}" \
        --overlap-words 2
      run "${OUT_ROOT}/icl_${tag}_big_seed_${seed}.json" \
        scripts/run_icl_baseline.py --model "${model}" --seed "${seed}" \
        --facts-per-task 8 --wide-labels
    done
  done
fi

coef_arm() {
  local name=$1 seed=$2; shift 2
  run "${OUT_ROOT}/coef_${name}_seed_${seed}.json" \
    scripts/run_controller.py --steps "${STEPS}" --seed "${seed}" \
    --no-gates --replay 1.0 --no-order-check "$@"
}

if has_part coef; then
  for seed in "${SEEDS[@]}"; do
    coef_arm d2_k8r4   "${seed}" --n-train-templates 2 --k 8  --rank 4
    coef_arm d4_k8r4   "${seed}" --n-train-templates 4 --k 8  --rank 4
    coef_arm d8_k8r4   "${seed}" --n-train-templates 8 --k 8  --rank 4
    coef_arm d2_k64r4  "${seed}" --n-train-templates 2 --k 64 --rank 4
    coef_arm d2_k32r16 "${seed}" --n-train-templates 2 --k 32 --rank 16
    coef_arm d8_k32r16 "${seed}" --n-train-templates 8 --k 32 --rank 16
  done
fi

if has_part lora; then
  for seed in "${SEEDS[@]}"; do
    for d in 2 8; do
      run "${OUT_ROOT}/lora_d${d}_seed_${seed}.json" \
        scripts/evaluate_sequence.py --steps "${STEPS}" --seed "${seed}" \
        --methods naive_stack --no-order-check --n-train-templates "${d}"
    done
  done
fi
set -u

echo "Done. Next: ${PYTHON_BIN} scripts/summarize_final.py"

#!/usr/bin/env bash
# Pressure grid: stress-test the replay retention result before trusting it.
# All arms controller-only, --no-gates, replay where noted. Every run now
# also records the cramming diagnostic (newest vector alone on earlier
# tasks). Artifacts land under artifacts/sweeps/pressure/ so
# summarize_sweeps.py labels them from config fields.
#
# Arms (see docs/roadmap_v0.2.md for the reasoning):
#   frac    replay=1 with 1/8, 1/4, 1/2 of earlier examples — does a sliver
#           of rehearsal repair composition, or only full joint training?
#   big     3 tasks x 8 facts, 12-label answer set, replay on/off — removes
#           the ceiling that compressed the retention grid to 1.00.
#   conflict overlap-words=2: same nonce words, conflicting labels across
#           domains, replay on/off — a composed state cannot satisfy these.
#   cap     k=2/4/16 with replay=1 (k=8 done) — does repair need capacity?
#
# Override lists from the shell:
#   SEEDS="0 1 2" STEPS=200 MODEL=distilgpt2 bash scripts/run_pressure.sh

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUT_ROOT="${OUT_ROOT:-artifacts/sweeps/pressure}"
STEPS="${STEPS:-200}"
MODEL="${MODEL:-distilgpt2}"
SEEDS=(${SEEDS:-0 1 2 3 4})

mkdir -p "${OUT_ROOT}"

echo "Writing pressure-grid artifacts under ${OUT_ROOT}"
echo "Python: ${PYTHON_BIN}  Model: ${MODEL}  Steps: ${STEPS}"
echo "Seeds: ${SEEDS[*]-}"

run_arm() {
  local outname=$1; shift
  local seed=$1; shift
  "${PYTHON_BIN}" scripts/run_controller.py \
    --model "${MODEL}" --steps "${STEPS}" --seed "${seed}" --no-gates "$@" \
    --out "${OUT_ROOT}/controller_${outname}_seed_${seed}.json"
}

set +u
for seed in "${SEEDS[@]}"; do
  # 1) replay-fraction sweep (default data)
  for frac in 0.125 0.25 0.5; do
    ftag="${frac//./p}"
    run_arm "frac_${ftag}" "${seed}" --replay 1.0 --replay-fraction "${frac}"
  done
  # 2) bigger family, ceiling removed: baseline + replay
  run_arm "big_replay_0"   "${seed}" --facts-per-task 8 --wide-labels
  run_arm "big_replay_1p0" "${seed}" --facts-per-task 8 --wide-labels --replay 1.0
  # 3) conflicting facts across domains: baseline + replay
  run_arm "conflict_replay_0"   "${seed}" --overlap-words 2
  run_arm "conflict_replay_1p0" "${seed}" --overlap-words 2 --replay 1.0
  # 4) capacity: k x replay (k=8 covered by the retention grid)
  for k in 2 4 16; do
    run_arm "cap_k${k}" "${seed}" --k "${k}" --replay 1.0
  done
done
set -u

echo
echo "Done. Aggregate with: python scripts/summarize_sweeps.py --stdout"

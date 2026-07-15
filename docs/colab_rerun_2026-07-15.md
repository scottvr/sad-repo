# Colab re-run queue — corrected coherence numbers (2026-07-15)

Purpose: regenerate every paraphrase/leakage column with the fixed probe
(commit `7a87d3d`; all historical coherence numbers measured the frozen
base — see CURRENT_STATUS "Coherence probe bug"). Retention, routing,
cramming, reversibility, and drift do NOT need re-running; only the
`coherence` block (and the new `coherence_base` floor) is new
information. Run stages in order — they're sorted by information value
per minute in case the session dies.

## Cell 0 — setup

```
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%%bash
git clone -b replay-scaling https://github.com/scottvr/sad-repo.git /content/sad-repo
pip -q install torch>=2.2 transformers>=4.40 pytest
mkdir -p /content/drive/MyDrive/sad-rerun-2026-07-15
```

Every stage below ends by syncing `artifacts/` to Drive so partial
results survive a disconnect.

## Stage 1 — validate the fix (~1 min, gates everything else)

```bash
%%bash
cd /content/sad-repo
python -m pytest tests -q
```

All tests must pass, in particular `tests/test_coherence_state.py`
(probe-time state pinned for all three methods). If these fail, stop.

## Stage 2 — eyeball raw outputs (~3 min)

```bash
%%bash
cd /content/sad-repo
python scripts/inspect_paraphrase.py --replay 1.0 --steps 200
cp -r artifacts /content/drive/MyDrive/sad-rerun-2026-07-15/
```

Read the printed table before trusting any aggregate: per-fact
restricted predictions + top-5 tokens for all 3 templates under
base / composed / routed states. What to look for:
- base state: template 0 and template 2 should show the constant,
  fact-independent label priors that produced the fake 0.0.
- composed state: does template 2 (never trained) ever get the right
  answer? This is the kill-criterion question, asked directly.
- routed state: same question with the task's own vector applied.

## Stage 3 — retention grid (headline replay=1 coherence) (~20 min)

```bash
%%bash
cd /content/sad-repo
bash scripts/run_retention.sh
cp -r artifacts /content/drive/MyDrive/sad-rerun-2026-07-15/
```

## Stage 4 — pressure grid (big/conflict/frac/capacity arms) (~25 min)

```bash
%%bash
cd /content/sad-repo
bash scripts/run_pressure.sh
cp -r artifacts /content/drive/MyDrive/sad-rerun-2026-07-15/
```

The big arm (24 facts, 12 labels) matters most here: base chance is
1/12, so a corrected paraphrase/leakage number has the most headroom to
move.

## Stage 5 — dims grid (~20 min)

```bash
%%bash
cd /content/sad-repo
bash scripts/run_dims.sh
cp -r artifacts /content/drive/MyDrive/sad-rerun-2026-07-15/
```

## Stage 6 — full multiseed suite (slowest; all four methods x 10 seeds)

```bash
%%bash
cd /content/sad-repo
bash scripts/run_multiseed.sh
cp -r artifacts /content/drive/MyDrive/sad-rerun-2026-07-15/
```

## Morning after

Copy the Drive `artifacts/` tree back over the repo's, then:

```bash
python scripts/summarize_sweeps.py --stdout
python scripts/summarize_multiseed.py --stdout
```

Interpretation notes (from CURRENT_STATUS):
- `coherence` is now the COMPOSED state; `coherence_base` is the floor.
  A composed paraphrase number is only meaningful relative to that floor.
- The probe compares template 0 (trained) vs 2 (held out), so it mixes
  consistency with generalization; `inspect_paraphrase.py` output is the
  disambiguator.
- Expected leakage floor: 1/6 = .167 (default labels), 1/12 = .083
  (wide). Anything at exactly the floor with zero variance across arms
  is the bug signature — if that reappears, distrust the run.

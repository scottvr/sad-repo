# sequential-adapt

Minimal experimental scaffold for one question:

> Can a frozen pretrained LM support **sequential, context-conditioned
> adaptation** via learned low-dimensional updates (coefficients over frozen
> random low-rank bases) and routing, such that multiple adaptations compose
> over time without catastrophic drift or semantic incoherence?

**Scope disclaimer:** this repo tests a *toy* version of that idea — a small
GPT-2 (distilgpt2), 3 synthetic fact domains, ~100-dim updates. A positive
result would only justify scaling to more realistic models/tasks. A negative
result may indicate bad implementation, bad task choice, or real instability.
It proves nothing beyond its own scale.

## Design in one paragraph

The base model (**Model A**) is frozen — verified by tests and runtime
checks. Every adapted site (attention + MLP input projections) holds K=8
frozen *random* rank-4 components; a task's update is just the coefficient
vector over those components (96 numbers for distilgpt2). Because updates are
linear in coefficients, composition is exact addition and reversal is exact
negation *in parameter space* — so any observed forgetting, order
sensitivity, or irreversibility is a property of behavioral composition
through the frozen network, not parameter bookkeeping. **Model B** is a small
MLP mapping frozen-base context embeddings → coefficient vectors. **Model C**
is per-site sigmoid gating fitted jointly with the coefficients.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; source .venv/bin/activate elsewhere
# GPU (CUDA 12.8 wheels; needed for RTX 50-series / sm_120):
pip install torch --index-url https://download.pytorch.org/whl/cu128
# or CPU-only:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Device is auto-detected (CUDA if available, else CPU); override with
`--device cpu|cuda` on any script. Everything is sized to run on CPU too —
the full suite is ~8 min CPU / ~1 min on an RTX 5070 Ti. Models download
from the HF Hub on first use (distilgpt2 ≈ 330 MB). Note: CPU and CUDA runs
differ slightly in low-level numerics, so exact accuracies can shift by one
probe item between devices; the qualitative results are stable.

## Run

```bash
python -m pytest tests -q            # ~5 s (uses a tiny random model)
python scripts/smoke_test.py         # ~2 min: independent + coeff_add + controller
python scripts/run_baselines.py      # + naive LoRA stacking, both task orders
python scripts/run_controller.py     # controller only; --ortho / --no-gates ablations
python scripts/evaluate_sequence.py  # everything (slowest)
```

Each script has `--help`; results are saved as JSON under `artifacts/`.

## Methods compared

| method | what it is |
|---|---|
| `independent` | one coefficient vector fitted per task, alone (upper bound; baseline 1) |
| `naive_stack` | per-task LoRA modules trained sequentially on top of each other, no control (baseline 2) |
| `coeff_add` | task-vector arithmetic: sum independently fitted coefficient vectors (baseline 3) |
| `controller` | sequential coefficient fitting with interference penalty + per-site gates; MLP controller routes coefficients from held-out context phrasings (proposed) |

## Metrics

Current-task acc/loss, retention matrix, forgetting, order sensitivity
(A→B→C vs C→B→A), reversibility (remove first task's update after the whole
sequence: return-to-base gap + collateral damage), drift (KL vs base on
neutral probes), coherence (paraphrase consistency + off-domain leakage).
Definitions: `src/sequential_adapt/metrics.py`, probes in `eval.py`.

## Findings so far (distilgpt2, seed 0 — read as smoke, not science)

- Single tasks fit perfectly in the 96-dim random-basis space (acc 1.0).
- **Composition is where it breaks**, as hypothesized: summing independently
  fitted vectors degrades all tasks; sequentially fitted coefficients are
  only valid in the presence of the earlier tasks' updates (applied alone
  they score near chance), so sequential fitting without control forgets
  early tasks.
- **Routing recovers it**: the MLP controller, given held-out phrasings of
  the domain context, predicts standalone-valid coefficients that restore
  0.75–1.0 of independent accuracy per task (1.0/1.0/1.0 in the GPU run).
- Exact-negation reversal of one task from an additive composition is clean;
  removing one task from a *sequentially* fitted stack causes large
  collateral damage on the rest.
- A KL-to-base drift anchor during fitting (`--anchor 1.0`) cuts
  neutral-probe drift ~12× (5.31 → 0.45) without hurting task fit, and makes
  routed accuracy perfect — but does not fix composed-sum interference.

Numbers: `artifacts/*.json` (`notes/experiment_plan.md` for the protocol,
`CURRENT_STATUS.md` for exact state and next commands).

## Layout

```
src/sequential_adapt/   config, data (synthetic fact domains), model (frozen
                        base), adapters (LoRA + shared random bases + bank),
                        controllers (MLP + lookup), train, eval, metrics,
                        experiments (orchestration)
scripts/                smoke_test, run_baselines, run_controller, evaluate_sequence
tests/                  frozen-base guarantees, adapter mechanics, metric math,
                        sequence-eval contract
notes/                  problem_statement, experiment_plan, literature_map_stub
artifacts/              JSON results (gitkeep'd)
```

# Scripts

Command-line entry points for running and summarizing the sequential
adaptation experiments. Run commands from the repo root.

## Quick Checks

```bash
python scripts/smoke_test.py
python scripts/evaluate_sequence.py --steps 200 --out artifacts/sequence_eval.json
python scripts/summarize_multiseed.py --stdout
```

`smoke_test.py` is the fastest sanity check. `evaluate_sequence.py` runs the
full single-seed comparison: independent, naive LoRA stack, coefficient
addition, and controller.

## Main Experiment Runners

| script | purpose |
|---|---|
| `smoke_test.py` | short CPU-friendly run for basic validation |
| `run_baselines.py` | baseline-focused comparison |
| `run_controller.py` | controller-only run with ablation flags |
| `evaluate_sequence.py` | full one-seed comparison, all methods |
| `run_multiseed.sh` / `run_multiseed.ps1` | 10-seed full-suite run plus anchored controller |
| `run_sweeps.sh` / `run_sweeps.ps1` | seed, K, ortho, no-gates, and task-count sweeps |
| `run_retention.sh` / `run_retention.ps1` | retention grid: {replay off/on} x {soft ortho / hard projection} |
| `summarize_multiseed.py` | aggregate `artifacts/sequence_seed_*.json` and `controller_anchor_seed_*.json` |
| `summarize_sweeps.py` | aggregate `artifacts/sweeps/**/*.json` |

Most Python runners accept:

```bash
--model distilgpt2
--steps 200
--seed 0
--device auto
--out artifacts/result.json
```

`evaluate_sequence.py` and `run_controller.py` also expose the main ablation
knobs:

```bash
--n-components 8    # alias: --k
--rank 4
--n-tasks 3
--facts-per-task 4
--ortho 0.1
--hard-ortho     # hard orthogonal projection vs earlier tasks (supersedes --ortho)
--anchor 1.0
--replay 1.0     # CE replay of earlier tasks' examples in the composed state
--no-gates
```

## Multi-Seed Run

Bash:

```bash
PYTHON=.venv/bin/python bash scripts/run_multiseed.sh
python scripts/summarize_multiseed.py --stdout
```

PowerShell:

```powershell
$env:PYTHON = ".\.venv\Scripts\python.exe"
.\scripts\run_multiseed.ps1
python scripts\summarize_multiseed.py --stdout
```

Default outputs:

```text
artifacts/sequence_seed_0.json
artifacts/controller_anchor_seed_0.json
artifacts/multiseed_summary.md
artifacts/multiseed_summary.csv
```

Override seeds or steps without editing the scripts:

```bash
SEEDS="0 1 2" STEPS=100 bash scripts/run_multiseed.sh
```

```powershell
.\scripts\run_multiseed.ps1 -Seeds @(0,1,2) -Steps 100
```

## Sweep Run

The sweep runner is intended for the larger GPU machine.

Bash:

```bash
PYTHON=.venv/bin/python bash scripts/run_sweeps.sh
python scripts/summarize_sweeps.py --stdout
```

PowerShell:

```powershell
$env:PYTHON = ".\.venv\Scripts\python.exe"
.\scripts\run_sweeps.ps1
python scripts\summarize_sweeps.py --stdout
```

Pilot run:

```bash
SEEDS="0 1" K_VALUES="8 16" ORTHO_VALUES="0 0.1" TASK_COUNTS="3 4" \
  bash scripts/run_sweeps.sh
```

```powershell
.\scripts\run_sweeps.ps1 `
  -Seeds @(0,1) `
  -KValues @(8,16) `
  -OrthoValues @(0,0.1) `
  -TaskCounts @(3,4)
```

Default sweep outputs land under:

```text
artifacts/sweeps/seed/
artifacts/sweeps/k/
artifacts/sweeps/ortho/
artifacts/sweeps/no_gates/
artifacts/sweeps/task_count/
artifacts/sweeps_summary.md
artifacts/sweeps_summary.csv
```

Task-count sweeps use `facts_per_task=4` up to 4 tasks and `facts_per_task=3`
above that, because the current nonce-word pool has 18 words.

## Retention Grid

Controller-only 2x2 grid over the two retention mechanisms: composed-state
replay (`--replay`) and hard orthogonal projection (`--hard-ortho`). The
`replay=0, no hard-ortho` arm is the baseline controller at the same seeds.
All arms run `--no-gates`: gates were inert in earlier sweeps, and hard
projection is only exact without them — this keeps every within-grid
comparison single-variable (cross-check against `artifacts/sweeps/no_gates/`).

```bash
PYTHON=.venv/bin/python bash scripts/run_retention.sh
python scripts/summarize_sweeps.py --stdout
```

```powershell
$env:PYTHON = ".\.venv\Scripts\python.exe"
.\scripts\run_retention.ps1
python scripts\summarize_sweeps.py --stdout
```

Pilot run:

```bash
SEEDS="0 1" REPLAY_VALUES="0 1.0" bash scripts/run_retention.sh
```

Outputs land under `artifacts/sweeps/retention/` and are picked up by
`summarize_sweeps.py` with condition labels like `replay=1;hard_ortho;controller`.

Key comparisons:

- Does replay lift the *composed* controller retention (the known failure)?
- Does hard projection beat the soft cosine^2 penalty on retention /
  collateral damage, and does it cost current-task accuracy?
- Do the two combined interact (replay shapes coefficients inside the
  projected subspace)?

## Colab Quickstart

Use [`colab_quickstart.ipynb`](../colab_quickstart.ipynb) at the repo root
(open via `https://colab.research.google.com/github/jmasseo/cc_fable_llm_lora_tests/blob/master/colab_quickstart.ipynb`).
It includes a GPU sanity check, clone/install, tests, smoke test, a pilot
retention grid, inline summary rendering, and artifact download/Drive
persistence (Colab storage is ephemeral). The equivalent shell commands:

```python
!git clone https://github.com/jmasseo/cc_fable_llm_lora_tests
%cd cc_fable_llm_lora_tests
!pip install -q transformers  # torch is preinstalled on Colab
!python -m pytest tests -q
!python scripts/smoke_test.py
!SEEDS="0 1 2" bash scripts/run_retention.sh
!python scripts/summarize_sweeps.py --stdout
```

## Interpreting Results

The summaries report mean +/- sample standard deviation across seeds. Key rows:

- `independent`: single-task upper bound, not a composed final state.
- `coeff_add`: naive additive composition of independently fitted vectors.
- `naive_stack`: sequential trainable LoRA modules.
- `controller`: composed shared-basis controller state.
- `controller_routed`: held-out context routing to standalone coefficients.
- `controller_anchor`: controller with KL-to-base drift anchoring.

Useful first comparisons:

- Does increasing `--n-components` improve retention without exploding drift?
- Does `--ortho` reduce order sensitivity or collateral damage?
- Does `--no-gates` hurt routed or composed controller behavior?
- Does drift anchoring reduce KL while preserving routed accuracy?
- Does task-count scaling fail gracefully or collapse suddenly?

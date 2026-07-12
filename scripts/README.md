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
--anchor 1.0
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

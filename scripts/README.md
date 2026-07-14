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
| `run_pressure.sh` / `run_pressure.ps1` | pressure grid: stress-test the replay result (frac/big/conflict/cap) |
| `run_dims.sh` / `run_dims.ps1` | dims grid: total coefficient dims vs site/layer allocation |
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
--sites attn     # both | attn | mlp — which projections get adapters
--layers 0-2     # restrict adapters to these transformer layers
--no-gates
```

Total coefficient dims = n_sites x k, so `--sites`/`--layers` vary dims
independently of `--k` (distilgpt2 default: 12 sites x 8 = 96).

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
above that. (Historical note: the pool was capped at 18 hand-picked nonce
words when those sweeps ran; a deterministic CVCV generator now extends it
past 300, so larger families are available via `--n-tasks`/`--facts-per-task`.)

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

Result (2026-07-14, 10 seeds): replay fixed composed retention completely
(0.36 -> 1.00), hard projection was inert, and replay's cost is surgical
reversibility. See `CURRENT_STATUS.md` — which is exactly why the next grid
exists.

## Pressure Grid

Stress-tests the replay result before trusting it (reasoning in
`notes/roadmap_v0.2.md`). Controller-only, `--no-gates`, 10 arms x 5 seeds:

- `frac`: replay=1 with `--replay-fraction` 0.125/0.25/0.5 — does a sliver
  of rehearsal repair composition, or only full joint training?
- `big`: 3 tasks x 8 facts, `--wide-labels` (12 answers), replay on/off —
  removes the ceiling that compressed the retention grid to 1.00.
- `conflict`: `--overlap-words 2` — the same nonce words carry *different*
  labels per domain; a single composed state cannot satisfy them all.
- `cap`: k=2/4/16 with replay=1 — does composed-state repair need
  coefficient capacity? (k=8 is covered by the retention grid.)

Every controller run now also records the **cramming diagnostic**
(`newest_alone_on_earlier`, summary column `cram (newest alone)`): the last
task's vector evaluated *alone* on earlier tasks' probes. High = the newest
vector re-learned everything (cramming); low = genuine composition repair.

```bash
PYTHON=.venv/bin/python bash scripts/run_pressure.sh
python scripts/summarize_sweeps.py --stdout
```

```powershell
$env:PYTHON = ".\.venv\Scripts\python.exe"
.\scripts\run_pressure.ps1
python scripts\summarize_sweeps.py --stdout
```

Pilot run:

```bash
SEEDS="0 1" bash scripts/run_pressure.sh
```

Outputs land under `artifacts/sweeps/pressure/` with condition labels
inferred from config fields (`replay=1;frac=0.25`, `replay=1;overlap=2`,
`replay=1;tasks=3x8;labels=12`, `replay=1;k=2`, ...).

## Dims Grid

Is 96 a magic number, or just `n_sites x k`? The pressure grid's `cap`
arms trace the both-sites dims curve (k=2/4/16 -> 24/48/192; 96 is the
retention grid's replay arm). This grid varies the *other* axis — which
sites get the budget — so the same dim count is realized by different
allocations. On distilgpt2, 48 dims appears five ways: `attn k=8`,
`mlp k=8`, `both k=4` (= pressure `cap_k4`), layers 0-2 `k=8`, layers 3-5
`k=8`. All arms `replay=1 --no-gates`.

- Retention/fit tracking total dims regardless of allocation → dims is the
  resource; 96 is nothing special.
- Allocation mattering (MLP vs attention, early vs late layers) → the
  story is *where* capacity sits, and any size/complexity scaling law must
  be sought per site type.

```bash
PYTHON=.venv/bin/python bash scripts/run_dims.sh
python scripts/summarize_sweeps.py --stdout
```

```powershell
$env:PYTHON = ".\.venv\Scripts\python.exe"
.\scripts\run_dims.ps1
python scripts\summarize_sweeps.py --stdout
```

Outputs land under `artifacts/sweeps/dims/` with labels like
`replay=1;k=2;sites=attn`, `replay=1;sites=mlp`, `replay=1;layers=0,1,2`.

## Colab Quickstart

Use [`colab/quickstart.ipynb`](../src/colab/quickstart.ipynb) at the repo root
(open via `https://colab.research.google.com/github/scottvr/sad-repo/blob/master/src/colab/quickstart.ipynb`).
It includes a GPU sanity check, clone/install, tests, smoke test, a pilot
retention grid, inline summary rendering, and artifact download/Drive
persistence (Colab storage is ephemeral). The equivalent shell commands:

```python
!git clone https://github.com/scottvr/sad-repo
%cd sad-repo
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

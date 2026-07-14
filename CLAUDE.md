# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A minimal research scaffold testing one hypothesis: can a frozen pretrained LM support sequential, context-conditioned adaptation via learned low-dimensional updates (coefficients over frozen random low-rank bases) plus routing, such that multiple adaptations compose without catastrophic forgetting or drift? It deliberately runs at toy scale (distilgpt2, 3 synthetic fact domains, ~96-dim updates). Results prove nothing beyond this scale — the README's scope disclaimer is intentional; keep it honest when editing docs.

## Commands

```bash
python -m pytest tests -q                # ~5 s, uses a tiny random model, no downloads
python -m pytest tests/test_adapters.py -q          # single test file
python -m pytest tests/test_adapters.py::test_name -q   # single test

python scripts/smoke_test.py             # fastest end-to-end sanity check (~2 min CPU)
python scripts/run_baselines.py          # baselines incl. naive LoRA stacking, both task orders
python scripts/run_controller.py         # controller only; --ortho / --anchor / --no-gates ablations
python scripts/evaluate_sequence.py      # full one-seed comparison, all methods (slowest)

bash scripts/run_multiseed.sh            # 10-seed suite (PowerShell: run_multiseed.ps1)
bash scripts/run_sweeps.sh               # k/ortho/gates/task-count sweeps (GPU machine)
bash scripts/run_retention.sh            # replay x hard-ortho retention grid (GPU machine)
bash scripts/run_pressure.sh             # replay stress tests: fraction/big-vocab/conflict/capacity (GPU machine)
bash scripts/run_dims.sh                 # dims grid: total coefficient dims vs site/layer allocation (GPU machine)
python scripts/summarize_multiseed.py --stdout
python scripts/summarize_sweeps.py --stdout
```

- Scripts import the package via `scripts/_bootstrap.py` (sys.path hack) — no install needed; run from repo root.
- All Python runners take `--model --steps --seed --device --out`; ablation knobs: `--n-components/--k --rank --n-tasks --facts-per-task --overlap-words --wide-labels --ortho --hard-ortho --anchor --replay --replay-fraction --sites --layers --no-gates`. Every script answers `--help`. Total coefficient dims = n_sites × k; `--sites {both,attn,mlp}` / `--layers 0-2` vary dims via site selection (`resolve_site_suffixes` in `config.py`), independently of `--k`.
- All results are saved as JSON under `artifacts/` — new experiments must follow this convention.
- Shell runners are parameterized by env vars (bash: `SEEDS="0 1" STEPS=100 bash scripts/run_sweeps.sh`) or params (PowerShell: `-Seeds @(0,1)`).
- CPU and CUDA runs differ by ±1 probe item due to numerics; qualitative results are stable. Device auto-detects.

## Architecture

Three cooperating "models" over one frozen base (all in `src/sequential_adapt/`):

- **Model A — frozen base** (`model.py`): distilgpt2 with all weights frozen; frozenness is verified by tests and runtime checks. Never unfreeze it.
- **Model B — coefficient controller** (`controllers.py`): MLP mapping frozen-base context embeddings → coefficient vectors (`MLPController`, with `LookupController` as deterministic fallback). Trained by regression against independently fitted coefficient targets, evaluated on held-out context phrasings.
- **Model C — gating** (`adapters.py` + `train.py`): per-site sigmoid gate logits fitted jointly with coefficients (`cfg.train_gates`).

Core mechanism (`adapters.py`): each adapted site (attention + MLP input projections) is wrapped in an `AdapterSite`. Two adapter kinds:

- `LoRAAdapter` — trainable A/B matrices, used only for the `naive_stack` baseline.
- `SharedBasisAdapter` — K frozen *random* rank-r components per site; a task's entire update is just a coefficient vector over them (~96 numbers). Because updates are linear in coefficients, composition is exact addition and reversal is exact negation in parameter space — any observed forgetting/order-sensitivity is a property of behavioral composition, not parameter bookkeeping. Preserve this linearity invariant when modifying adapters.

`AdapterBank` (`adapters.py`) owns per-task coefficient tensors shared across sites; `bank.apply([(task, sign), ...])` rebinds live references, and coefficients are read lazily at forward time so optimizer updates are always reflected.

Sequential fitting (`train.py`): each new task's coefficients are optimized with earlier tasks' updates active but frozen. Loss = CE + L2 + cosine² penalty vs. earlier tasks' coefficient directions (`ortho_penalty`) + optional KL-to-base drift anchor on neutral probes (`anchor_weight`) + optional composed-state replay CE on earlier tasks' examples (`replay_weight`). `hard_ortho` replaces the soft penalty with projected gradient descent: after each step the new task's raw coefficients are projected orthogonal to earlier tasks' directions (exact with gates off).

`experiments.py` orchestrates the four compared methods: `independent` (upper bound), `naive_stack`, `coeff_add` (task-vector arithmetic), `controller` (proposed). Metric definitions (retention matrix, forgetting, order sensitivity, reversibility, drift KL, coherence) live in `metrics.py`; probes in `eval.py`. Central `Config` dataclass in `config.py`.

Tasks (`data.py`) are synthetic nonce-word → color mappings in 3 domains, 3 templates per fact (2 trained, 1 held out for routing eval). The nonce pool starts with the original 18 hand-picked words (kept first so older runs reproduce exactly) and extends deterministically via a CVCV generator; `--wide-labels` switches to a 12-color answer set, and `--overlap-words N` makes the first N words of every domain shared with conflicting labels (adversarial for composed states).

## Findings so far (read before designing new experiments)

- Single tasks fit perfectly in the shared random basis; **composition by summing fails hard** — even sequential fitting with the ortho penalty forgets early tasks.
- **Routing (selective application) is what works**: the MLP controller recovers full independent accuracy from held-out phrasings.
- Sweeps showed k, gating, and ortho-penalty strength have little useful effect at this scale — don't re-tune them.
- Drift anchoring (`--anchor 1.0`) cuts neutral-probe KL ~10× without hurting task fit, but does not fix composed-sum interference.
- **Composed-state replay (`--replay 1.0`) fixed composed retention completely** (0.36 → 1.00 over 10 seeds) at no cost to current-task fit or routing; its price is surgical reversibility (vectors become co-adapted). Hard orthogonal projection was inert — coefficient-space orthogonality is a dead end at this scale. The replay result is **not yet pressure-tested** (cramming/ceiling/conflict controls: `run_pressure.sh`); don't build on it until that grid runs. Kill-criterion and rationale: `notes/roadmap_v0.2.md`.
- Exact state, headline numbers, and next commands: `CURRENT_STATUS.md`. Protocol: `notes/experiment_plan.md`. Original brief: `INSTRUCTIONS.md`.

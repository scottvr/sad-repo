# Current status

Everything described in the README runs, on CPU and on the local
RTX 5070 Ti (torch 2.11.0+cu128; device auto-detected, `--device` to
override). No known failures.

## Verified (commands actually run on 2026-07-08, Windows, Python 3.13)

```
.venv/Scripts/python -m pytest tests -q          -> 19 passed (~4 s)
.venv/Scripts/python scripts/smoke_test.py       -> artifacts/smoke_results.json
                                                    (12.7 s GPU / ~95 s CPU)
.venv/Scripts/python scripts/evaluate_sequence.py --steps 200
                                                 -> artifacts/sequence_eval.json
                                                    (60 s GPU / ~500 s CPU)
.venv/Scripts/python scripts/run_controller.py --steps 200 --anchor 1.0 \
    --out artifacts/controller_anchored.json     -> (69 s GPU)
```

All scripts answer `--help`. Current artifacts are from the GPU runs; CPU
runs reproduce the same qualitative results with ±1-probe accuracy shifts
(different low-level numerics).

## Headline numbers (distilgpt2, seed 0, 200 steps, tasks A→B→C, GPU run)

Final accuracy per task (A/B/C), measured on a trained phrasing:

| method | A | B | C | avg retention | rev gap | collateral | order sens | drift KL |
|---|---|---|---|---|---|---|---|---|
| base | .25 | .25 | .00 | – | – | – | – | 0 |
| independent (per-task) | 1.0 | 1.0 | 1.0 | – | – | – | – | 3.9 |
| naive LoRA stack | .75 | .75 | 1.0 | .75 | .25 | .25 | .25 | 7.6 |
| coefficient addition | .25 | .00 | .00 | .12 | .00 | .00 | .00 | 6.2 |
| controller (composed sum) | .25 | .50 | 1.0 | .38 | .25 | .62 | .58 | 4.6 |
| controller (**routed**) | 1.0 | 1.0 | 1.0 | – | – | – | – | – |
| controller + drift anchor (composed) | .25 | .25 | 1.0 | .25 | .00 | .62 | .58 | **0.46** |
| controller + drift anchor (**routed**) | 1.0 | 1.0 | 1.0 | – | – | – | – | – |

Interpretation (toy scale — do not over-read):

1. Any single task fits perfectly in the 96-dim frozen random basis.
2. **Composition in a shared basis fails hard.** Summing independently
   fitted coefficient vectors interferes so strongly that even the most
   recent task breaks. Sequential fitting (with earlier updates active)
   protects the newest task but forgets the oldest; the cosine²
   interference penalty (0.1) did not save it.
3. **Separate parameter subspaces compose better**: naive LoRA stacking
   (fresh matrices per task) retains 0.75 — but removing one module from the
   stack is dirty (0.25 collateral), because later modules were trained in
   its presence.
4. **Routing wins at this scale**: an MLP mapping frozen-base context
   embeddings → independently fitted coefficients recovers full independent
   performance from held-out context phrasings. Selective application, not
   superposition, is what worked.
5. **The drift anchor works on drift, not interference**: `--anchor 1.0`
   cuts neutral-probe KL ~10× (4.6 → 0.46) with no loss of current-task fit
   and perfect routing, but composed-sum retention stays poor.
6. Paraphrase consistency is 0.0 everywhere: 96-dim updates memorize the
   trained phrasing and do not transfer to an unseen template. Low-dim
   updates at this scale are surface-level, not semantic.

## New, implemented but NOT yet run (2026-07-13, written on a no-GPU/no-torch
## dev machine — syntax-checked only, needs verification on the CUDA box/Colab)

Multi-seed runs and the k/ortho/gates/task-count sweeps are done; they showed
k, gating, and ortho-penalty strength don't move the needle. Two retention
mechanisms were added before scaling the model:

1. **Composed-state replay** (`--replay W`, `cfg.replay_weight`): while
   fitting task N with earlier updates active, add `W * CE` on earlier
   tasks' training examples — directly penalizes the new coefficients for
   breaking earlier behavior *in composition* (the known failure mode).
2. **Hard orthogonal projection** (`--hard-ortho`, `cfg.hard_ortho`):
   projected gradient descent — after every optimizer step the new task's
   raw coefficients are projected onto the orthogonal complement of earlier
   tasks' directions (QR basis, flat [n_sites*K] space; only 1 dim lost per
   earlier task out of ~96). Supersedes the soft cosine² penalty. Exact with
   `--no-gates`; approximate when the new task's own gates are trained.

Touched: `config.py`, `train.py` (`fit_task_coefficients`), `experiments.py`
(`run_controller` passes `replay_tasks`), `run_controller.py`,
`evaluate_sequence.py`, `run_retention.sh/.ps1`, `summarize_sweeps.py`,
`tests/test_retention.py`.

## Suggested first commands on the GPU box / Colab

```
python -m pytest tests -q                 # includes new test_retention.py
python scripts/smoke_test.py              # regression check
python scripts/run_controller.py --steps 200 --replay 1.0 --hard-ortho \
    --out artifacts/controller_retention_probe.json   # single probe run
bash scripts/run_retention.sh             # full 2x2 grid x 10 seeds
python scripts/summarize_sweeps.py --stdout
```

## Older next steps still open

- Gradient-free coefficient fitting (CMA-ES over ~100 dims).
- End-to-end controller training (current pipeline distills independently
  fitted coefficients into the MLP; the distillation bottleneck may hide
  whether routing generalizes).
- Scale model (gpt2 → TinyLlama) and task count — AFTER the retention grid.

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

## What remains / next steps (in value order)

1. **Multi-seed runs with error bars** — everything above is one seed.
   Cheap now (~1 min/run on the GPU).
2. **Ablations**: `--ortho` sweep (0 / 0.1 / 1.0), `--no-gates`, K
   (n_components) sweep 4→32 — does a bigger shared basis reduce
   interference? Scriptable today via existing flags except K (edit config).
3. Replay-free retention losses beyond cosine² (e.g. penalize change in
   earlier tasks' training logits — needs a small activation cache).
4. Gradient-free coefficient fitting (CMA-ES over ~100 dims) to make
   "Model B predicts updates without backprop" literal.
5. Scale model (gpt2 → TinyLlama, now feasible on the 5070 Ti) and task
   count (n_tasks up to 6 works with the current word/domain pools).

## Suggested first command for a human

```
python scripts/smoke_test.py
```

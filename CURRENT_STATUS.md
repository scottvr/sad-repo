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

## Retention grid: mechanisms + results (implemented 2026-07-13, run
## 2026-07-14 on Colab, RTX 6000 Blackwell; artifacts under
## artifacts/sweeps/retention/, aggregated in artifacts/retention_summary.md)

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

### Results (composed controller, 10 seeds, all arms --no-gates)

| arm | final acc | retention | order sens | rev gap | collateral |
|---|---|---|---|---|---|
| baseline (retention_ctrl) | .58 ± .09 | .36 ± .14 | .58 | .05 | .68 |
| hard_ortho | .54 ± .07 | .31 ± .11 | .62 | .03 | .64 |
| **replay=1** | **1.00 ± .00** | **1.00 ± .00** | **.00** | .05 | **.89** |
| replay=1 + hard_ortho | .98 ± .05 | .98 ± .08 | .02 | .05 | .88 |

Routed accuracy unaffected in all arms (0.98–1.0).

Interpretation:

7. **Replay fixes composed-sum forgetting completely** at this scale:
   retention 0.36 → 1.00 with zero seed variance, order sensitivity
   0.58 → 0.00, no cost to current-task fit or routing. Notably, the repair
   is expressed through only the *new* task's ~96 trainable coefficients —
   the composed function is repaired from inside a tiny frozen random basis.
8. **Coefficient-space orthogonality is a dead end.** Exact hard projection
   moves nothing (all deltas within noise), alone or on top of replay —
   consistent with the earlier soft-penalty sweeps. The interference is not
   in coefficient geometry; it lives in function space, which replay
   attacks directly. Hypothesis "isolated subspaces enable interference-free
   composition" is falsified at this scale (soft AND hard variants).
9. **Replay's cost is surgical reversibility**: collateral 0.68 → 0.89.
   Replay-fitted vectors are co-adapted — task N's coefficients encode
   "task N + repairs for the composed state," so exact negation of one
   vector no longer cleanly removes just that task. The exact-arithmetic
   add/remove property (the original selling point of linear composition)
   is traded for retention. Note collateral was already poor (0.68) at
   baseline, so what was sacrificed was mostly already broken.

Touched: `config.py`, `train.py` (`fit_task_coefficients`), `experiments.py`
(`run_controller` passes `replay_tasks`), `run_controller.py`,
`evaluate_sequence.py`, `run_retention.sh/.ps1`, `summarize_sweeps.py`,
`tests/test_retention.py`.

## Open questions raised by the retention grid

- **Is replay's win trivial?** Rehearsal is the oldest fix in continual
  learning; "replay prevents forgetting" is expected. The non-trivial parts
  here are (a) the repair fits in ~96 dims of a frozen random basis, and
  (b) the reversibility price is exactly measurable because composition is
  linear. Controls worth running: replay with a *fraction* of earlier
  examples (does tiny replay suffice?); joint fitting of all tasks at once
  (upper bound — does sequential+replay match it?).
- **Does the repair capacity survive more tasks?** With T tasks the newest
  vector must repair T-1 earlier tasks. Task-count sweep with replay=1 is
  the natural next grid (the old 18-nonce-word cap is gone — see below).
- **Can reversibility be bought back?** E.g. decompose each task's vector
  into a pure part (fit independently) + an explicit interaction-repair
  part (fit with replay), so removal deletes pure+repair terms together
  and bookkeeping stays exact. (That is the `reversible-composition`
  branch's question; this branch is `replay-scaling`.)

## Pressure grid: implemented, NOT yet run (2026-07-15, replay-scaling
## branch, written on the no-torch dev machine — syntax-checked only)

Everything needed to stress-test the replay result (rationale and
kill-criterion: `notes/roadmap_v0.2.md`):

1. **Cramming diagnostic** (automatic in every controller run): each
   sequentially fitted vector is also evaluated ALONE; JSON field
   `newest_alone_on_earlier`, summary column `cram (newest alone)`.
   High = the newest vector re-learned earlier tasks (12 facts fit easily
   in ~96 dims); low = genuine composition repair. **Read this column
   before believing any retention number.**
2. **`--replay-fraction F`**: deterministic per-task subsample of replay
   examples (>= 1 per earlier task). Separates "a sliver of rehearsal
   suffices" from "full joint training in disguise".
3. **Bigger task family**: nonce pool now extends past 300 via a CVCV
   generator (original 18 words kept first — historical task sets are
   byte-identical); `--wide-labels` = 12 single-token colors;
   `--overlap-words N` = shared words with conflicting labels across
   domains (a composed state cannot satisfy them; routing should).
4. **`scripts/run_pressure.sh` / `.ps1`**: 10 arms x 5 seeds (~50 runs,
   ~20 min on a fast card) — frac 0.125/0.25/0.5, big family (8 facts x 12
   labels) with replay on/off, conflict (overlap=2) with replay on/off,
   capacity k=2/4/16 with replay. Artifacts: `artifacts/sweeps/pressure/`.

### First commands on Colab / GPU box

```
python -m pytest tests -q                 # includes new data + diagnostic tests
python scripts/run_controller.py --steps 200 --replay 1.0 --no-gates \
    --out artifacts/pressure_probe.json   # single probe: check the cram column
bash scripts/run_pressure.sh              # full pressure grid
python scripts/summarize_sweeps.py --stdout
```

### How to read the outcome

- `cram (newest alone)` HIGH (≈ earlier-task accuracy) in `replay=1` arms →
  replay is cramming, not repair; the 1.00 retention headline deflates.
- `frac=0.125` retention ≈ 1.0 → tiny rehearsal suffices (interesting);
  only `frac=1` works → joint-training-in-disguise (expected/dull).
- `big` arms: retention off ceiling; the replay-vs-baseline *gap* is the
  real effect size.
- `conflict` arms: composed accuracy on overlapped facts is capped by
  construction (~(1 - overlap/facts) ceiling on the composed state);
  routed accuracy should stay high. If composed retention still reads
  ~1.0 here, be suspicious of the eval, not pleased.
- `cap` arms: retention 1.0 at k=2 (24 dims) → repair is cheap, capacity
  story weak; retention degrading with k → capacity relationship exists
  (then find the scaling law before TinyLlama).

## Dims grid: implemented, NOT yet run (same session, syntax-checked only)

Is 96 a magic number, or just `n_sites * k`? New `--sites {both,attn,mlp}`
and `--layers 0-2` flags on `run_controller.py` / `evaluate_sequence.py`
vary total coefficient dims via site selection, independently of `--k`
(`resolve_site_suffixes` in `config.py`; torch-free tests in
`tests/test_sites.py`). `scripts/run_dims.sh/.ps1` (7 arms x 3 seeds,
all replay=1 --no-gates): an attention-only dims curve (12/24/48/96) plus
five different realizations of a 48-dim budget (attn k=8, mlp k=8, both
k=4 via pressure `cap_k4`, layers 0-2, layers 3-5). If performance tracks
total dims regardless of allocation, dims is the resource; if allocation
matters, the size/complexity relationship is per-site-type. Artifacts:
`artifacts/sweeps/dims/`. Run this after the pressure grid — priority is
deflating/confirming the replay headline first.

## Older next steps still open

- Gradient-free coefficient fitting (CMA-ES over ~100 dims).
- End-to-end controller training (current pipeline distills independently
  fitted coefficients into the MLP; the distillation bottleneck may hide
  whether routing generalizes).
- Scale model (gpt2 → TinyLlama) and task count — retention grid is done;
  scaling is now unblocked, with replay=1 (no hard-ortho) as the default
  retention mechanism.

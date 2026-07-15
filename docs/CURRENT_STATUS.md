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
6. ~~Paraphrase consistency is 0.0 everywhere: 96-dim updates memorize the
   trained phrasing and do not transfer to an unseen template. Low-dim
   updates at this scale are surface-level, not semantic.~~ **RETRACTED
   2026-07-15: measurement bug.** Every historical coherence number was
   computed after `reset_adapters()`, i.e. on the frozen base model (see
   "Coherence probe bug" section below). Whether adapted states transfer
   to unseen templates is UNKNOWN pending re-run.

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

## Pressure grid: RUN 2026-07-14 (GPU box; artifacts under
## artifacts/sweeps/pressure/, single probe in artifacts/pressure_probe.json)

Everything needed to stress-test the replay result (rationale and
kill-criterion: `docs/roadmap_v0.2.md`):

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

### Results (composed controller, 5 seeds/arm, all --no-gates)

| arm | retention | cram (newest alone) | current fit | routed |
|---|---|---|---|---|
| frac=0.125 | .35 ± .10 | .18 | 1.00 | .98 |
| frac=0.25 | .45 ± .17 | .13 | 1.00 | .98 |
| frac=0.5 | .55 ± .14 | .10 | 1.00 | .98 |
| frac=1 (retention grid, 10 seeds) | 1.00 ± .00 | – | 1.00 | .98 |
| big (24 facts, 12 labels), no replay | .11 ± .08 | .13 | 1.00 | .91 |
| big, replay=1 | .69 ± .09 | .05 | .88 | .91 |
| conflict (overlap=2), no replay | .20 ± .07 | .23 | 1.00 | .98 |
| conflict, replay=1 | .98 ± .06 | .13 | 1.00 | .98 |
| cap k=2 (24 dims), replay=1 | .35 ± .10 | .28 | .30 | .40 |
| cap k=4 (48 dims), replay=1 | .70 ± .11 | .23 | .75 | .90 |
| cap k=16 (192 dims), replay=1 | 1.00 ± .00 | .25 | 1.00 | .98 |

Verdicts (numbering continues the interpretation list above):

10. **Cramming refuted.** `newest_alone_on_earlier` is at or below the
    base-model floor (.05–.28) in every arm — the newest vector does not
    re-learn earlier tasks; replay genuinely repairs the composition.
11. **Rehearsal fraction: the textbook result.** Retention rises roughly
    linearly with fraction and saturates only at full rehearsal —
    mechanically this is joint training with earlier vectors frozen. The
    non-trivial residue is *where* the repair lives (~96 dims of a frozen
    random basis, per verdict 10). Pseudo-replay (rehearsal text generated
    by the composed model itself) is now the important escape hatch.
12. **The ceiling was real.** On the big family replay lifts retention
    0.11 → 0.69, not to 1.0, and current-task fit slips to 0.88. Headline
    correction: replay repairs composition *completely at 12 facts,
    partially at 24*.
13. **Conflict is a genuine pass.** Retention probes include the
    conflicting facts (verified in `data.py`), the prompt template names
    the domain, and the composed replay-fitted state disambiguates:
    0.20 → 0.98. This is the context-conditioning the original hypothesis
    wanted — but note the domain is named *explicitly* in the prompt, so
    it is a shallow version of the ability (see roadmap v0.3 confounds).
14. **There is a capacity floor and it scales with facts.** k=2 (24 dims)
    breaks single-task fit (.30) and even routing (.40); k=4 (48 dims) is
    partial everywhere; 96 dims saturates 12 facts but only reaches 0.69
    on 24 facts. Same law as the dims grid (below): ~4 dims/fact ≈ 0.7
    retention, ~8 dims/fact ≈ 1.0.

~~Paraphrase consistency is 0.000 in every arm, again — the kill-criterion
clock (`docs/roadmap_v0.2.md`) is unchanged by any of this.~~ **RETRACTED
2026-07-15 — see "Coherence probe bug" below; these numbers measured the
frozen base, not the adapted model.**

## Dims grid: RE-RUN 2026-07-15 (GPU box; artifacts under
## artifacts/sweeps/dims/; supersedes the 2026-07-14 console-only run,
## whose parsed table remains in artifacts/sweeps/dims_console_summary.csv)

Is 96 a magic number, or just `n_sites * k`? `--sites {both,attn,mlp}` and
`--layers 0-2` vary total coefficient dims via site selection,
independently of `--k` (`resolve_site_suffixes` in `config.py`).
8 arms x 5 seeds, all replay=1 --no-gates (7 allocation arms at 12 facts,
plus the `big_k16` prediction arm: 24 facts, 12 labels, both sites k=16);
joined below with the pressure/retention both-sites arms at matching
total dims:

| allocation | dims | facts | retention | composed fit | routed |
|---|---|---|---|---|---|
| attn k=2 | 12 | 12 | .30 ± .11 | .23 | .23 |
| attn k=4 | 24 | 12 | .30 ± .07 | .28 | .67 |
| both k=2 (pressure) | 24 | 12 | .35 ± .10 | .33 | .40 |
| attn k=8 | 48 | 12 | .72 ± .14 | .75 | 1.00 |
| mlp k=8 | 48 | 12 | .60 ± .10 | .58 | .88 |
| layers 0-2 k=8 | 48 | 12 | .70 ± .21 | .68 | .90 |
| layers 3-5 k=8 | 48 | 12 | .72 ± .10 | .77 | .98 |
| both k=4 (pressure) | 48 | 12 | .70 ± .11 | .72 | .90 |
| attn k=16 | 96 | 12 | 1.00 ± .00 | 1.00 | .97 |
| both k=8 (retention) | 96 | 12 | 1.00 ± .00 | 1.00 | .98 |
| both k=16 (pressure) | 192 | 12 | 1.00 ± .00 | 1.00 | .98 |
| big_k16: both k=16 | 192 | 24 | 1.00 ± .00 | 1.00 | .90 |

Cram (`newest_alone_on_earlier`) is at or below the base floor in every
arm (big_k16: .09 vs. a .125 floor), so none of this is the newest vector
re-learning earlier tasks.

Verdicts:

15. **Total dims is the resource; 96 is nothing special.** The attn-only
    curve overlays the both-sites curve at matched dims; attention-only at
    96 dims is as perfect as both-sites. The five 48-dim allocations sit
    in one band (.60–.72, within a std of each other at n=5); MLP-only
    trails slightly if anything. No "where capacity sits" story survives.
16. **Dims-per-fact prediction CONFIRMED (out of sample).** ~4 dims/fact
    ≈ 0.7 retention, ~8 dims/fact saturates — consistent across this grid
    (48/96 dims, 12 facts) and the pressure big arm (96 dims, 24 facts →
    0.69). The pre-registered test: `big_k16` doubles dims to 192 on the
    same 24-fact family and should restore retention to ~1.0. Result:
    1.00 ± .00 retention and 1.00 composed fit over 5 seeds, cram at the
    floor. Still two family sizes and one model — a robust local law,
    not yet a scaling law.
17. Routing saturates earlier than retention (~0.9–1.0 at 48 dims where
    retention is ~0.7) — routing is the easier problem, consistent with
    every earlier grid.

## Coherence probe bug (FOUND 2026-07-15, fixed in code, re-run pending)

**Every paraphrase-consistency and off-domain-leakage number recorded
before 2026-07-15 is invalid: the probe measured the frozen base model.**
In `experiments.py`, all three sequence methods called `reset_adapters()`
(after the reversibility / routed evals) before `_finalize()`, which is
where `coherence_probe` ran. Proof from the artifacts, no re-run needed:
at a fixed seed, coherence is byte-identical across every arm — k=2 vs
k=16, replay on/off, attn/mlp/layer allocations — and off-domain leakage
is exactly `1/len(label_space)` (0.1667 for 6 labels, 0.0833 for 12),
the base model's restricted-argmax chance rate. Paraphrase consistency
0.0 with zero variance was the base model deterministically preferring
different label priors under template 0 vs template 2 ("...maps to the
color" vs "...the color associated with {w} is").

Consequences:

- The kill-criterion (roadmap v0.2: "paraphrase consistency > 0 never
  appears") has never actually been tested. Its status is UNKNOWN, not
  failing. Verdict 6 above and the pressure-grid paraphrase line are
  retracted.
- Retention/routing/cramming/reversibility/drift numbers are unaffected
  (different code paths, probed in the correct states — the arm-to-arm
  variance in those metrics is itself the evidence).
- Note template asymmetry when re-running: templates 0 and 1 are trained,
  2 is held out; the probe compares 0 vs 2, so consistency conflates
  "same answer" with "generalizes to an untrained phrasing". Report
  trained-vs-trained (0 vs 1) alongside 0-vs-2 to separate those.

Fix (2026-07-15): callers now restore the composed state before
`_finalize`, which probes coherence and only then resets; the base floor
is recorded separately as `coherence_base` in every result. Regression
tests: `tests/test_coherence_state.py`.

## Corrected coherence numbers: RE-RUN 2026-07-15 (Colab; all grids)

All suites re-run under the fixed probe (retention, pressure, dims,
multiseed + raw-output dump `artifacts/paraphrase_inspection.json`).
First sanity: every previously reported retention/routing/cramming
number reproduces exactly (replay 0.36 → 1.00, big 0.69, big_k16 1.00),
so the fix perturbed nothing else. The corrected coherence columns:

| arm | paraphrase | leakage |
|---|---|---|
| multiseed controller (replay=0) | .12 ± .13 | — |
| multiseed naive_stack (full-rank LoRA) | **.88 ± .08** | — |
| multiseed coeff_add | .63 ± .19 (artifact: agreeing-wrong) | — |
| retention replay=1 | .19 ± .11 | .77 ± .15 |
| retention replay=0 | .23 ± .17 | .46 ± .10 |
| pressure cap_k16, replay=1 | .25 ± .06 | **.97 ± .04** |
| pressure big replay=1 (12 labels) | .07 ± .05 | .42 ± .09 |
| pressure conflict replay=1 | .23 ± .09 | .39 ± .07 |
| dims big_k16 | .10 ± .06 | .69 ± .17 |
| base floor (`coherence_base`) | 0.00 | .17 (=1/6) |

Verdicts:

18. **The kill-criterion signal is real, but now honestly measured:
    coefficient updates show ~chance paraphrase transfer.** Raw dump
    (replay=1, seed 0): trained templates 12/12 and 12/12, held-out
    template 1/12. Composed consistency (.07–.25) sits at the label-
    chance agreement rate (~.17 for 6 labels, ~.08 for 12). Not the
    fake structural 0.0 anymore — but no transfer either.
19. **NEW — full-rank LoRA transfers where 96-dim coefficients don't.**
    naive_stack paraphrase consistency is .88 with good trained-template
    fit, vs .12 for the coefficient controller in the same runs. Caveat:
    consistency conflates "agrees" with "correct" (coeff_add scores .63
    while wrong everywhere — same wrong prior under both templates), so
    this needs the new `final_evals_heldout` accuracy metric (added
    2026-07-15, in every future run) to confirm. If it holds, paraphrase
    transfer is gated by update capacity/rank, not model scale — which
    would redirect the roadmap: the cheap test is raising k/rank, not
    TinyLlama.
20. **NEW — composed states are word-keyed, not context-conditioned.**
    Off-domain leakage: prompting a domain-A word inside domain-B's
    template yields A's label 42–97% of the time in composed states
    (1.00 in the raw dump), scaling with fit quality, vs a .17 chance
    floor. With disjoint nonce words the domain prompt is simply
    ignored; only the conflict arm (shared words force context use,
    leakage .39) behaves context-conditionally. "Context-conditioned
    adaptation" currently rests entirely on routing — the composed
    state itself is a word→label lookup smeared across the bases.

## Next phases

Confound ledger (label-bias null, foil probes, routing-shortcut and
memorization diagnostics) and the distilgpt2 → gpt2 → TinyLlama scaling
arcs are planned in `docs/roadmap_v0.3.md`. Arc 0's first command
(`bash scripts/run_dims.sh`) ran 2026-07-15: dims JSONs are collected at
5 seeds and the `big_k16` dims-per-fact prediction is confirmed (above).

**Reprioritized 2026-07-15 after the corrected coherence numbers** (full
decision ledger: roadmap v0.3): next GPU session runs the **caprank
grid** (`bash scripts/run_caprank.sh` — does held-out-template accuracy
rise with k/rank at fixed scale?) plus a multiseed rerun to populate
`final_evals_heldout` for the naive_stack-vs-controller comparison. The
Colab notebook (`src/colab/sad-quickstart.ipynb`) is set up for exactly
this. TinyLlama (Arc 3) is off the critical path until caprank answers;
end-to-end controller training is promoted (routing is the only
context-conditioned part of the system, per verdict 20 / confound C9).

## Older next steps still open

- Gradient-free coefficient fitting (CMA-ES over ~100 dims).
- End-to-end controller training (current pipeline distills independently
  fitted coefficients into the MLP; the distillation bottleneck may hide
  whether routing generalizes).
- Scale model (gpt2 → TinyLlama) and task count — retention grid is done;
  scaling is now unblocked, with replay=1 (no hard-ortho) as the default
  retention mechanism. gpt2 needs zero code changes; the Llama-family
  blockers are cataloged in `docs/tinyllama_readiness.md` (the silent
  BOS-as-gold-label bug is already fixed; site-suffix/layer-prefix
  mapping and a tokenizer label audit remain).

# Experiment plan

## Task family

Synthetic fact domains. Domain X holds `facts_per_task` nonce-word → color
facts (e.g. domain A: blicket→red, dax→blue …; domain B uses different nonce
words and a shifted label pattern). Prompts come from 3 templates:

- T0, T1: training phrasings ("In domain A, the word blicket maps to the color")
- T2: held-out phrasing, never trained on.

Answers are single-token color words; **accuracy = argmax restricted to the
6-color label space**, so it is meaningful even when the base model is weak.
Accuracy/retention is measured on T0 (a *trained* phrasing — the question is
whether learned behavior survives later adaptation, not template
generalization). T2 feeds the paraphrase-coherence probe.

Deliberately unrealistic: the purpose is to expose composition failure modes
cheaply, not to model real domains.

## Methods

| method       | update space                  | sequence protocol                                   |
|--------------|-------------------------------|-----------------------------------------------------|
| independent  | shared-basis coefficients     | each task fitted alone from base (upper bound)      |
| naive_stack  | per-task LoRA modules         | trained sequentially, each atop the previous, no control |
| coeff_add    | shared-basis coefficients     | independently fitted vectors summed (task arithmetic) |
| controller   | shared-basis coefficients + gates | fitted sequentially with earlier tasks applied, cosine² interference penalty, per-site sigmoid gates (Model C); plus MLP context→coefficients (Model B) for routed eval |

Shared basis: K=8 frozen random rank-4 components at each of 12 sites
(distilgpt2: `attn.c_attn` and `mlp.c_fc` in all 6 blocks) → a task is a
96-dim coefficient vector.

## Metrics (all implemented in `metrics.py` / `eval.py`)

- current task acc/loss (restricted argmax, CE)
- retention matrix R[i][j]: acc on task j after stage i
- forgetting: immediate-after acc − final acc, per task
- order sensitivity: forward vs reversed sequence, mean |Δacc|
- reversibility: after the full sequence, remove the first task's update
  (negate coefficients / disable LoRA module); report return-to-base gap on
  that task and collateral |Δacc| on the others
- drift: mean KL(base ‖ adapted) of final-position logits on 8 neutral probes
- coherence: paraphrase consistency (T0 vs T2 predictions agree) and
  off-domain leakage (fact answered with its label under the wrong domain)

## Protocol

1. `python scripts/smoke_test.py` — distilgpt2, 80 steps, independent +
   coeff_add + controller, forward order only (~2 min CPU).
2. `python scripts/run_baselines.py` — baselines incl. naive LoRA stacking,
   both orders.
3. `python scripts/run_controller.py` — controller method, both orders;
   `--ortho`, `--no-gates` for ablations.
4. `python scripts/evaluate_sequence.py` — everything, one JSON.

All results land in `artifacts/`.

## Ablations worth running next (not automated yet)

- `--ortho 0` vs `0.1` vs `1.0`: does the interference penalty buy retention?
- `--no-gates`: does Model C matter at all at this scale?
- more tasks (`n_tasks` up to 6 supported by the word/domain pools)
- K (n_components) sweep: 4 → 32; retention vs expressivity trade-off
- drift-anchored fitting (add KL-to-base term to the fitting loss) — not
  implemented; the hook would go in `train.fit_task_coefficients`.

## Known limitations (deliberate, v0)

- 3 tasks × 4 facts; statistics of one seed. No error bars.
- The MLP controller sees only ~15 (context, coefficient) pairs; it memorizes.
  The held-out *phrasing* evaluation is the only generalization check.
- Coefficient fitting backprops through the frozen base (activation grads);
  a truly gradient-free variant (e.g. CMA-ES over 96 dims) is future work.
- Base model is distilgpt2; nothing here says results transfer upward.

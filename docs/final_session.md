# Final session: pre-registered stop/continue test

Written and committed 2026-09-30, **before any of these numbers exist**.
The git timestamp on this file is the pre-registration. The thresholds below
are fixed; `scripts/summarize_final.py` applies them mechanically and prints
the verdict. If the verdict is STOP, this document plus that summary is the
last section of the write-up.

## Why this session exists

Every result so far reproduces something already known (random-basis
adapters fit single tasks; summed task vectors interfere; routing beats
merging; replay prevents forgetting; fact fine-tuning on 2 phrasings does
not transfer to new phrasings). The project has never faced the obvious
competitor: **just put the facts in the prompt.** An adapter method for
injecting facts is only worth pursuing if it beats that on something.

This session asks exactly two questions, both on held-out phrasings (the
only accuracy that means "learned the fact", not "memorized a string"):

1. **Usefulness.** Do tiny coefficient adapters beat prompting the same
   frozen model with the same facts?
2. **Transfer lever.** If adapter held-out accuracy moves at all, is it
   more training phrasings (data diversity, the known lever) or more
   update capacity (k, rank; the question caprank was built for)?

## The metric

**Held-out-pool accuracy**: restricted-argmax accuracy over the label
space, averaged over every fact × 4 phrasings that are never trained in
any arm (templates 2, 9, 10, 11 in `data.TEMPLATES`). Chance is 1/6 ≈ .17.
The base model with no facts is the floor (ICL condition `base`).

Paired comparisons (same frozen distilgpt2, same 12 facts, 3 domains):

| adapter condition | prompting analogue |
|---|---|
| **composed**: all 3 tasks' vectors active at once (controller, replay=1) | **icl_all**: all 12 facts in context |
| **routed**: controller-selected single vector | **icl_domain**: only that domain's 4 facts in context |

## Arms

All adapter arms: distilgpt2, 200 steps, replay=1, `--no-gates`,
`--no-order-check`, seeds 0–4. `dN` = N training phrasings.

| arm | train phrasings | k | rank | coefficient dims | role |
|---|---|---|---|---|---|
| `d2_k8r4` | 2 | 8 | 4 | 96 | anchor (= every historical run) |
| `d2_k64r4` | 2 | 64 | 4 | 768 | capacity: components |
| `d2_k32r16` | 2 | 32 | 16 | 384 (richest bases) | capacity: rank |
| `d4_k8r4` | 4 | 8 | 4 | 96 | diversity |
| `d8_k8r4` | 8 | 8 | 4 | 96 | diversity |
| `d8_k32r16` | 8 | 32 | 16 | 384 | both levers at once |

References (not eligible for a CONTINUE verdict, reported for context):

- **LoRA** (`naive_stack`, trainable rank-4 A/B matrices) at d2 and d8.
  If only LoRA transfers, the continuation would be "use LoRA", which is
  not this project.
- **Prompting** on distilgpt2, gpt2 (124M) and gpt2-medium (355M); for
  each: default family, conflict family (2 shared words with conflicting
  labels per domain), big family (24 facts, 12 labels). Seeds shuffle the
  order facts are stated in. The decision uses the default family; the
  others are reported.

Every controller arm also records reversibility (collateral damage to the
other tasks when one task's vector is negated), so the cost of replay on
clean removal is reported alongside, for the reversible-composition
question.

## Decision rule (fixed now)

Let, on the default family, seed means:

- `A_comp` = best composed held-out-pool accuracy over the six adapter
  arms, counting only arms with composed retention ≥ 0.90
- `A_route` = best routed held-out-pool accuracy over the six adapter arms
- `P_all(m)`, `P_dom(m)` = prompting held-out-pool accuracy of model `m`
  under icl_all / icl_domain

A pairing (`A_comp` vs `P_all`, or `A_route` vs `P_dom`) **wins** iff all
three hold:

1. adapter ≥ **0.60** (clearly learned the facts, not just above chance)
2. adapter − P(distilgpt2) ≥ **0.15** (beats prompting the same model by
   a margin larger than 5-seed noise)
3. adapter − P(gpt2-medium) ≥ **0.15**: any continuation would move to
   larger models, where prompting only gets better. If prompting a 355M
   model already catches up, the adapter's edge disappears exactly where
   the continuation would have to live.

Taking the best of six arms inflates the adapter number a little
(selection on noise); the 0.15 margins are deliberately wider than that.

**CONTINUE (narrowed)** iff at least one pairing wins. The narrowed scope
would be "fact injection at zero prompt tokens", re-tested on a model
where prompting is strong (e.g. Qwen-0.5B-class), with the winning lever
(diversity or capacity) as the recipe.

**STOP** otherwise, labelled by the first matching reason:

- **prompting wins**: in both pairings, P(distilgpt2) ≥ adapter − 0.05.
- **edge vanishes with scale**: some pairing passes conditions 1 and 2
  but fails 3.
- **no transfer**: every adapter arm, composed and routed, has held-out
  accuracy ≤ 0.30. The updates store strings, not facts, whatever their
  capacity or training diversity.
- **inconclusive**: anything else. **Ties go to stopping**: this session
  was declared the last unless it shows something clearly worth pursuing.

Informational (does not change the verdict): lever attribution, i.e.
diversity effect = `d8_k8r4` − `d2_k8r4` versus capacity effect =
max(`d2_k64r4`, `d2_k32r16`) − `d2_k8r4`, on routed held-out accuracy.

## Predictions (written before running)

- Prompting on distilgpt2: icl_domain well above chance, icl_all lower
  (12 facts is a lot of context for an 82M model); both rise with model
  size.
- Adapters at d2: held-out accuracy near chance regardless of k/rank
  (capacity effect small).
- Adapters at d8: clearly higher; diversity effect larger than capacity
  effect. This is the known result that fact learning needs varied
  phrasings.
- Predicted verdict: **STOP**, most likely "prompting wins" or "edge
  vanishes with scale". The live route to CONTINUE is prompting being
  surprisingly weak at 82M–355M on nonce facts while d8 adapters transfer.

## How to run

Colab: `src/colab/sad-quickstart.ipynb`, section "Final session". Or:

```bash
bash scripts/run_final.sh          # ~1.5-2 h on a T4; SEEDS="0 1" for a pilot
python scripts/summarize_final.py  # tables + mechanical verdict
```

Artifacts: `artifacts/final/`; summary: `artifacts/final_summary.md` and
`artifacts/final_summary.json`.

## Open thread: reversible composition (pure + repair)

Not part of the decision rule; it doesn't affect the verdict. Written
so it reads as a closing note if the verdict is STOP, or as the pickup
point if the project continues.

**The property.** Each task's update is linear in its coefficient
vector, so adding a task and removing it are exact in parameter space.
Linearity itself is not new (random-basis adapters and task-vector
arithmetic are linear too). The unusual part here is using it as a
measuring instrument. Because the parameter side of a removal is exact
by construction, any damage seen after removing a task is purely
behavioral, and it can be measured cleanly. Most unlearning methods
can't separate the two.

**What was measured.** Removing one task by negating its vector damages
the remaining tasks. Collateral is .68 without replay and .89 with
replay (10 seeds, retention grid). Replay makes vectors co-adapted:
task N's vector encodes "task N plus repairs for the composed state".
Every controller arm in this session records collateral again, so its
cost is reported next to the transfer numbers.

**The idea, untested.** Split each task into a *pure* vector (fit
alone) plus explicit *interaction-repair* vectors (fit with replay, one
per pair of tasks). To remove a task, delete its pure vector and every
repair term that involves it. The bookkeeping stays exact, and every
term is still a tiny coefficient vector.

**What would make it worth a project of its own:**

1. **A real need for one merged state.** Keeping adapters separate and
   routing between them already gives perfect removal: stop applying
   the one you want gone. Pure + repair only matters when everything
   must live in a single summed state, for example folded into the
   weights for serving.
2. **Collateral near zero after removal**, with retention still ≥ .9
   before removal. This is the first experiment: 3 tasks, remove each
   in turn, compare against plain negation (.89) and routing (0).
3. **Tolerable growth.** Repair terms grow with the number of task
   pairs (quadratically), so a task-count sweep has to show storage
   staying small or repairs staying sparse.

**Related work to position against:** negating task vectors to make a
model forget (task arithmetic), and exact unlearning by retraining on
data shards (SISA). The angle neither covers: provable removal from a
merged adapter state.

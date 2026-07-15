# Roadmap after v0.2.0 (written 2026-07-15, `replay-scaling` branch)

Decisions and criteria agreed in conversation but not previously captured
in the repo. The retention-grid results this builds on are in
`docs/CURRENT_STATUS.md`; the original protocol is `docs/experiment_plan.md`.

## Kill-criterion (written down deliberately)

**If generalization beyond trained phrasings (paraphrase consistency > 0)
has not emerged by TinyLlama scale with the richer vocabulary, re-scope the
goal from "general sequential adaptation" to fact-injection / exact-recall
personalization / unlearning.** Paraphrase consistency is 0.0 in every
condition so far: ~96-dim updates memorize surface forms. That may be a
scale artifact (distilgpt2, 12 facts) or it may be fundamental to
low-dimensional linear updates. Either answer is publishable; pretending
the question isn't open is not.

## Why the replay result must be pressure-tested before celebrating

Replay=1.0 gave composed retention 1.00 ± 0.00 (baseline 0.36). Reasons to
distrust perfection, in order of severity:

1. **Cramming hypothesis**: total corpus = 12 facts; one ~96-dim vector has
   capacity for all of them. Replay may not be "repairing composition" —
   the newest vector may simply re-learn every earlier task. *Diagnostic:*
   evaluate the newest task's vector **alone** (earlier vectors removed) on
   earlier tasks' probes. High accuracy = cramming, and the retention
   result is far less interesting.
2. **Full rehearsal = joint training in disguise**: replay weight 1.0 over
   *all* earlier examples ≈ joint optimization with earlier vectors frozen.
   *Control:* replay-fraction sweep (0.1 / 0.25 / 0.5 of earlier examples).
   If a sliver of rehearsal suffices, the mechanism is interesting; if it
   needs everything, it's the textbook result.
3. **Ceiling effects**: 3 tasks × 4 facts, handful of probes — perfect
   scores compress all differences. *Fix:* grow nonce pool (18 → ~60),
   facts per task (4 → 8–10), and the answer-class set, keeping runtime in
   the same class.
4. **Task homogeneity**: every task is nonce→color with identical format.
   *Adversarial case:* shared nonce words across domains with conflicting
   answers ("blarp is red" in A, "blarp is blue" in C). Replay cannot
   satisfy both in one composed state unless context genuinely
   disambiguates — this is the case the original hypothesis was supposed
   to handle.

## The "96" question

96 is not magic: it is `n_sites × k` = (6 layers × 2 sites) × 8. It scales
with depth × k, so TinyLlama (22 layers, attn+MLP) at k=8 ≈ 352 dims.
Open question: does the capacity *needed* (for fit and for replay-repair)
scale with task count, facts per task, model size — or none of these? The
earlier k-sweep (4/8/16/32, inert) was **pre-replay**; a k × replay grid is
the capacity experiment now that the composed state actually retains.
Possible outcomes: repair capacity ~ facts (cramming), ~ tasks (interaction
terms), or ~ constant (true low-dim structure).

## Falsified / settled (do not revisit without new evidence)

- Coefficient-space orthogonality (soft cosine² penalty AND exact hard
  projection) does nothing for composed retention at this scale. The
  interference lives in function space, not coefficient geometry.
- k (4–32), gating, ortho-weight: inert pre-replay; only k is worth
  re-testing, and only in combination with replay (capacity question).
- MoE-style mixture composition: rejected as against the spirit of the
  project (single composed state, explicit vectors).

## Branch strategy (from v0.2.0)

- **`replay-scaling`** (active, this branch): pressure-test replay, grow
  data, capacity grid, then scale model. Accepts entanglement; goal is a
  single merged state serving all tasks.
- **`reversible-composition`**: buy back surgical reversibility (replay's
  cost: collateral 0.68 → 0.89). Sketch: decompose each task into a *pure*
  vector (fit independently) + an explicit *interaction-repair* vector
  (fit with replay); removal deletes pure + its repair terms together, so
  bookkeeping stays exact. Storage grows with interactions but every term
  is still a tiny coefficient vector.
- Cross-pollination expected; findings that fit the other branch better
  move there.

## Why any of this would matter (viability yardstick)

If the end-state works (frozen base + per-task few-hundred-float vectors):

- **Adapter economics**: ~400 bytes/task vs MBs for LoRA; thousands of
  users/domains per GB; adapter switching at effectively zero serving
  latency (the problem S-LoRA/Punica engineer around).
- **Exact unlearning** (reversible branch's prize): provable removal by
  negation — regulatory (GDPR-style deletion) and safety-rollback value
  that "retrain and hope" cannot offer.
- **Auditability**: a behavior expressed as ~100 inspectable numbers.

**Yardstick:** quality per adapter-byte and per training-FLOP vs rank-1
LoRA on the same base. Concrete bar: at TinyLlama scale, on a sequential
multi-domain benchmark, ≥90% of LoRA quality at ~1/1000 the storage.

**Known threats to the story:**

- Paraphrase consistency 0.0 (see kill-criterion) — caps the niche at
  exact-recall use cases unless it improves with scale.
- Replay requires **retaining old task data**: O(T) compute per new task
  and a privacy conflict in exactly the personalization settings where the
  economics are best. Escape hatch to keep on the roadmap: pseudo-replay
  (generate rehearsal text from the composed model itself).

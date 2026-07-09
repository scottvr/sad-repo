# Problem statement

## Hypothesis

A frozen pretrained language model can support **sequential, context-conditioned
adaptation** via learned low-dimensional updates — coefficient vectors over a
fixed bank of frozen random low-rank bases — such that multiple adaptations
compose over time **without catastrophic drift or semantic incoherence**, and
such that individual adaptations remain **selectively reversible**.

Three components:

- **Model A** — frozen pretrained base LM (never updated).
- **Model B** — update controller: maps a context representation (frozen-base
  embedding of the current domain) to a low-dimensional coefficient vector,
  not a full gradient or full LoRA delta.
- **Model C** — routing/gating controller: per-site gates deciding which
  layers receive which coefficient strengths.

## Why coefficients over frozen random bases (the key design choice)

Each adapter site holds K frozen random rank-r components; a task's update is
only the K-vector of coefficients per site (~10^2 numbers total, vs ~10^5 for
LoRA on the same sites). Consequences:

1. **Composition is exact vector addition in parameter space.** Applying tasks
   A+B means adding coefficient vectors. There is no bookkeeping ambiguity.
2. **Reversal is exact negation in parameter space.** So any failure to revert
   *behavior* is necessarily a statement about composition through the frozen
   network (task B's update was fitted in the presence of A's), not about
   sloppy parameter arithmetic. The scaffold separates these two things,
   which full-LoRA stacking cannot.
3. The update space is small enough that a tiny MLP (Model B) can plausibly
   learn the map context → coefficients from a handful of examples.

## What counts as success

On the synthetic fact-domain tasks (see `notes/experiment_plan.md`), with the
frozen base and the proposed controller:

1. **Retention:** after adapting to tasks A→B→C, accuracy on A and B stays
   near their immediately-after-training values (forgetting ≲ 0.1 absolute
   accuracy), materially better than naive stacking / naive addition.
2. **Routing works:** the MLP controller, given a *held-out phrasing* of a
   domain context, predicts coefficients that recover ≥ 0.75 of the
   independently-fitted accuracy on that domain.
3. **Reversibility:** negating one task's coefficients after the full
   sequence returns that task to base-level behavior (gap ≲ 0.1) with
   collateral damage on other tasks ≲ 0.1.
4. **Bounded drift:** KL(base‖adapted) on neutral probes for the composed
   model is not much worse than for a single adaptation.

## What counts as failure

- Composed accuracy collapses to chance on earlier tasks no matter the
  penalty/gating configuration (catastrophic interference is intrinsic at
  this scale).
- The controller cannot separate domains from frozen-base context embeddings
  (routing no better than applying an average coefficient vector).
- Reversal of one task destroys the others (updates are entangled, not
  modular).
- Drift on neutral probes grows multiplicatively with each added task.

A negative result here does **not** falsify the general idea — it may reflect
a bad task family, an underpowered base model, or the random-basis choice —
but it would mean the simplest version does not survive contact with reality,
and any scaled-up claim needs a different mechanism.

## Why this is not merely "LoRA exists"

LoRA answers: *can a low-rank update specialize a frozen model to one task?*
(Yes; settled.) This repo holds that constant and asks the questions LoRA
does not answer:

- **Sequential composition:** what happens to task A's behavior when B's and
  C's updates are added later, in the same shared low-dimensional space?
- **Retention vs interference:** is forgetting in coefficient space as severe
  as in weight space? Do interference penalties (coefficient orthogonality)
  or per-layer gating change that?
- **Reversibility:** LoRA modules can be detached, but *stacked* sequentially
  trained modules were each fitted in the context of the previous ones —
  does clean detachment survive that? (Baseline 2 tests exactly this.)
- **Context-conditioned routing:** can "which update to apply" itself be
  predicted from the frozen model's own representation of the context,
  making adaptation self-directed rather than externally switched?

Related prior work exists for each piece in isolation — task-vector
arithmetic (Ilharco et al.), LoRAHub / AdapterFusion (learned combination of
pretrained adapters), intrinsic-dimension fine-tuning (Aghajanyan et al.,
random subspace training), hypernetwork-generated adapters, and the continual
learning literature (EWC, orthogonal gradient methods). See
`notes/literature_map_stub.md`. The under-explored intersection this repo
targets: **sequential** adaptation in a **shared frozen random basis** with
**exact-arithmetic reversibility accounting** and a **frozen-base-driven
router**, measured with retention/forgetting/order/reversibility/drift/
coherence on one controlled task family.

## The actual hard problem

The hard problem is not fitting any single task (trivially achievable — the
independent baseline hits 100%). It is that a *sequence* of updates through a
shared nonlinear function interferes:

- fitting task B while A is applied produces coefficients for B that are only
  valid *in A's presence* (observed directly: sequentially fitted
  coefficients applied alone perform poorly);
- adding independently fitted updates degrades all of them (observed:
  coefficient addition drops accuracy toward chance);
- removing one update from a sequentially fitted stack breaks the rest
  (observed: high collateral in the controller reversibility probe).

Retention, reversibility, order-invariance, and bounded drift are therefore
*joint* constraints that the naive mechanisms already fail at toy scale.
The open question this scaffold makes measurable: how much of that failure
can cheap control (routing, gating, interference penalties) buy back.

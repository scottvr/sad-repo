# Literature map (stub)

Pointers only — written from memory during an offline sprint; verify
citations before relying on them. The point of this file is to mark where
this repo's question sits relative to known work, and to be honest about
what is already done.

## Low-dimensional adaptation of frozen models

- **LoRA** (Hu et al., 2021) — low-rank weight deltas on a frozen base.
  Settled; this repo's baseline substrate.
- **Intrinsic dimensionality of fine-tuning** (Aghajanyan et al., 2020; Li &
  Liang random-subspace work) — fine-tuning succeeds in a random
  low-dimensional reparameterization. Direct ancestor of this repo's
  "coefficients over frozen random low-rank bases" design.
- **(IA)³ / prefix / prompt tuning** — even lower-dimensional per-task
  updates; no sequential-composition story.

## Combining adaptations

- **Task arithmetic / task vectors** (Ilharco et al., 2022) — adding and
  negating fine-tuning deltas. Baseline 3 is exactly this, in coefficient
  space. Known to interfere; interference at toy scale reproduced here.
- **AdapterFusion** (Pfeiffer et al., 2021) — learned attention over a set of
  pretrained adapters. Simultaneous, not sequential; adapters are big.
- **LoRAHub** (Huang et al., 2023) — learned scalar mixing of pretrained
  LoRAs for a new task. Closest published cousin of "Model B predicts
  coefficients"; mixing is per-new-task and offline, not context-conditioned
  routing at inference.
- **Model merging line** (TIES, DARE, etc.) — resolving interference when
  summing deltas. Same failure mode as observed here, attacked at merge time
  rather than at fit time.

## Continual learning

- **EWC / SI / orthogonal-gradient methods (OGD, GPM)** — protecting earlier
  tasks during later training. The cosine² penalty on coefficient vectors
  here is a (much cruder) coefficient-space analogue of GPM's idea.
- **Progressive networks, per-task adapters** — sidestep interference by
  never sharing update space; the independent baseline is the degenerate
  version.

## Routing / control

- **Hypernetworks for adapters** (Karimi Mahabadi et al., 2021) — a network
  generates adapter params from a task embedding. Model B is a minimal
  hypernetwork whose output space is deliberately tiny (≈100 dims) and whose
  input is the frozen base's own context embedding rather than a learned
  task token.
- **MoE routing** — token-level routing among experts trained jointly. Here
  routing is post-hoc over post-trained updates, base frozen.

## The gap this repo probes

Each ingredient exists. The specific combination that seems under-measured:
**sequential** adaptation in a **shared frozen random basis**, where
composition/reversal are exact in parameter space (isolating behavioral
interference), evaluated jointly on retention, order sensitivity,
reversibility, drift, and coherence, with routing driven by the frozen
model's own context representation. If a published paper already does
exactly this, the honest next step is to reproduce it here and reduce this
repo to a replication + metric suite.

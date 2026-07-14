Treat this as a high-agency overnight research-engineering sprint.

GOAL
Build the smallest viable research repo for testing this hypothesis:

Can a frozen pretrained model support sequential, context-conditioned adaptation via learned low-dimensional updates and/or routing, such that multiple adaptations compose over time without catastrophic drift or semantic incoherence?

This is not a production library. It is a rigorous experimental scaffold designed to expose whether the idea survives first contact with reality.

CORE IDEA
We want a minimal implementation of:

- Model A: frozen pretrained base model.
- Model B: update controller that predicts low-dimensional adapter coefficients, not full gradients.
- Model C: routing/gating controller that decides which layers/adapters receive which coefficient strengths.
- Sequential adaptation: apply adaptation A, then B, then C, then test whether earlier behavior survives.


HARD CONSTRAINTS
- Keep the repo runnable on modest hardware.
- Prefer tiny Hugging Face models first, e.g. distilgpt2, gpt2, TinyLlama, or another small causal LM.
- Do not require giant training runs.
- Use PyTorch.
- Use PEFT/LoRA only if it simplifies things; otherwise implement a tiny custom low-rank adapter wrapper.
- Base model weights must remain frozen.
- Favor measurable results over architectural cleverness.
- No speculative giant framework.
- No half-finished abstractions.
- Write tests.
- Every script should have --help.
- Every generated result should be saved under artifacts/.
- If something is too large to complete, implement the smallest truthful version and document what remains.

DELIVERABLES
Create a repo with roughly this structure, adjusting only if there is a better simple design:

README.md
requirements.txt or pyproject.toml
src/
  sequential_adapt/
    __init__.py
    config.py
    data.py
    model.py
    adapters.py
    controllers.py
    train.py
    eval.py
    metrics.py
    experiments.py
scripts/
  smoke_test.py
  run_baselines.py
  run_controller.py
  evaluate_sequence.py
tests/
  test_adapters.py
  test_metrics.py
  test_sequence_eval.py
artifacts/
  .gitkeep
notes/
  problem_statement.md
  experiment_plan.md
  literature_map_stub.md

PROBLEM STATEMENT DOC
Write notes/problem_statement.md containing:

- Crisp hypothesis.
- What counts as success.
- What counts as failure.
- Why this is not merely "LoRA exists".
- Why the actual hard problem is sequential composition, retention, reversibility, and drift.

MINIMAL TASK FAMILY
Pick or create a synthetic task family that is cheap but meaningful.

Acceptable first version:
- Prompt-classification or next-token preference tasks using a tiny frozen LM.
- Synthetic "facts" or label mappings, e.g.:
  - In domain A, "blicket -> red"
  - In domain B, "dax -> blue"
  - In domain C, "wug -> green"
- The model is adapted sequentially to these small domains.
- Evaluation probes whether adaptations compose, interfere, or reverse.

Do not over-optimize dataset realism. The first purpose is to expose composition failure modes.

BASELINES
Implement at least these:

1. Independent adapter per task:
   - Train or fit one adapter per task.
   - Evaluate each independently.

2. Naive sequential adapter stacking:
   - Apply adapters A then B then C without learned routing/control.
   - Evaluate retention after each step.

3. Task-vector/adapter coefficient addition:
   - Combine adapter deltas or coefficients by simple addition/averaging where possible.

4. Proposed controller:
   - Fixed low-rank bases.
   - Learned or fitted coefficients per task/context.
   - Optional learned routing/gating over layers.
   - If full learning is too much, implement a simple trainable MLP/controller and a deterministic fallback controller.

METRICS
Implement these metrics:

- Current task accuracy or loss.
- Retention:
  performance on previous tasks after later adaptations.
- Forgetting:
  immediate-after-task score minus score after later tasks.
- Order sensitivity:
  compare A->B->C against C->B->A.
- Reversibility:
  apply A, then an approximate inverse or negative coefficient, and test return-to-base behavior.
- Drift:
  distance between base logits and post-sequence logits on neutral probes.
- Coherence:
  simple contradiction or consistency checks over synthetic fact probes.

EXPERIMENT RUNNER
Implement a script that runs a complete small experiment:

python scripts/smoke_test.py

It should:
- Load tiny model.
- Freeze base weights.
- Build adapters/controllers.
- Create tiny synthetic tasks.
- Run at least one baseline and one controller method.
- Print a concise metrics table.
- Save JSON results to artifacts/smoke_results.json.

Then implement:

python scripts/run_baselines.py
python scripts/run_controller.py
python scripts/evaluate_sequence.py

They may call common library functions.

TESTING
Add pytest tests for:
- Base weights remain frozen.
- Adapter changes output.
- Adapter can be enabled/disabled.
- Sequential evaluation returns expected keys.
- Retention/forgetting metrics are numerically sane.
- Reversibility metric runs without crashing.

SELF-VERIFICATION
Before stopping:
- Run tests.
- Run smoke_test.py.
- If failures remain, fix them if feasible.
- If not feasible, write CURRENT_STATUS.md explaining exactly what works, what fails, and the next command the human should run.

RESEARCH RIGOR
Avoid pretending this proves the full idea. The README should clearly say:

This repo tests a toy version of sequential low-dimensional adaptation. A positive result would only justify scaling to more realistic models/tasks. A negative result may indicate bad implementation, bad task choice, or real instability.

STYLE
- Be direct.
- Prefer simple code.
- Avoid huge abstractions.
- No decorative prose.
- No fake progress.
- Ground all progress claims in actual files, commands, and test results.

AUTONOMY INSTRUCTIONS
When you have enough information to act, act.
Do not stop to ask questions.
Make reasonable choices and document them.
If a dependency is unavailable, choose a simpler fallback.
If GPU is unavailable, keep everything CPU-runnable.
If you have extra time/tokens, choose a high value refinement or feature of your own devising and go for it. 
NB: if this idea is no longer novel due to other papers/implementations that have come out since I wrote this, modify this plan so that it tries to prove something novel and beneficial, by a small but meaningful revision to this plan.
If time is running out, prioritize:
1. runnable smoke test
2. metrics
3. baselines
4. controller
5. README/status docs

FINAL RESPONSE
At the end, report:
- files created/modified
- commands run
- test results
- experiment results
- what remains incomplete
- the most promising next step
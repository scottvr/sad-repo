# TinyLlama readiness audit (2026-07-15, replay-scaling branch)

What must change to scale `distilgpt2 -> gpt2 -> TinyLlama`. Audit of
GPT-2 assumptions in `src/`; no code changed yet. gpt2/gpt2-medium need
**nothing** (same architecture; `--layers` and site suffixes just cover
12/24 layers instead of 6) — they are the free intermediate step.

## Blockers for Llama-family models

1. **BOS prepended by `encode()`** — the biggest and quietest bug.
   `tokenizer.encode(a)[0]` is used for gold/leak label ids at
   `eval.py:23`, `eval.py:78`, `train.py:15`. Llama tokenizers prepend
   BOS, so `[0]` would be the BOS id for every label — training and eval
   would silently target BOS. Fix: `add_special_tokens=False` everywhere
   a label is encoded (and in `check_single_token_labels`, `data.py:165`,
   which would otherwise raise on len==2 — loud, but the fix is the same).

2. **Label space may not be single-token.** SentencePiece treats leading
   spaces differently (` red` -> `▁red`, usually one token, but verify all
   6 + 12 labels). `check_single_token_labels` already fails loudly at
   runtime; run it against the TinyLlama tokenizer *first* and swap any
   multi-token color for a single-token one.

3. **Site suffixes are GPT-2 names.** `SITE_GROUPS` (`config.py`) maps to
   `attn.c_attn` / `mlp.c_fc`; Llama blocks are
   `self_attn.{q,k,v,o}_proj` / `mlp.{gate,up,down}_proj`. Needs a
   family-aware `SITE_GROUPS` (e.g. keyed off `model.config.model_type`),
   and a decision: which Llama projections are the analogue of the fused
   `c_attn` (q_proj+v_proj is the standard LoRA choice) and of `c_fc`
   (up_proj). Adapter math is fine: `AdapterSite` adds `adapter(x)` to the
   module output, and `_in_out_dims` (`adapters.py:26`) already duck-types
   `nn.Linear` vs Conv1D.

4. **`--layers` prefix is `h.{i}`** (`resolve_site_suffixes`,
   `config.py`). Llama layer paths are `model.layers.{i}.`; the prefix
   must come from the same family mapping as (3).

## Things verified fine

- Pad-token fallback to EOS (`model.py:10`) works for Llama (also lacks
  a pad token). Last-non-pad indexing (`model.py:60`, `train.py:21`)
  assumes right padding — both tokenizers default to right padding; do
  not flip to left for generation-style batching.
- `context_embedding` (`model.py:66`) uses `output_hidden_states` mean —
  architecture-agnostic; controller `d_model` is read from the model.
- Freezing + `attach_adapter_sites` are name-agnostic (suffix matching).
- Nonce prompts may tokenize to multiple tokens per word — already true
  for CVCV words under GPT-2; nothing assumes one token per word.

## Suggested order of work (after the pressure grid runs)

1. gpt2 run with current code (`--model gpt2`) — zero changes, tests the
   depth axis (24 sites -> 192 dims at k=8; ties into the dims grid).
2. `add_special_tokens=False` fix + a tokenizer-portability test.
3. Family-aware `SITE_GROUPS` / layer prefix + TinyLlama label audit.

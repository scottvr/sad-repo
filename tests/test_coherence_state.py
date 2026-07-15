"""Regression tests for the coherence-probe state bug (fixed 2026-07-15).

Every result recorded before the fix computed coherence_probe AFTER
reset_adapters(), i.e. measured the frozen base model — paraphrase
consistency was identically 0.0 and off-domain leakage exactly
1/len(label_space) in every arm. These tests pin the contract: the final
coherence probe must run in the composed (adapted) state, and the base
floor is reported separately as `coherence_base`.
"""

import sequential_adapt.experiments as exp
from sequential_adapt.experiments import (run_coeff_addition, run_controller,
                                          run_naive_stack)


def _spy_probe(ctx, monkeypatch, seen):
    real = exp.coherence_probe

    def spy(model, tokenizer, tasks, cfg):
        seen.append({
            "applied": list(ctx.bank.applied),
            "lora_enabled": [
                sum(1 for n, on in site.enabled.items()
                    if on and n != "shared")
                for site in ctx.sites.values()
            ],
        })
        return real(model, tokenizer, tasks, cfg)

    monkeypatch.setattr(exp, "coherence_probe", spy)


def test_coeff_addition_probes_composed_state(ctx, monkeypatch):
    seen = []
    _spy_probe(ctx, monkeypatch, seen)
    res = run_coeff_addition(ctx, list(ctx.tasks))
    assert len(seen) == 1
    assert len(seen[0]["applied"]) == len(ctx.tasks), (
        "coherence probe ran without the composed coefficients applied")
    assert "coherence_base" in res


def test_controller_probes_composed_state(ctx, monkeypatch):
    seen = []
    _spy_probe(ctx, monkeypatch, seen)
    res = run_controller(ctx, list(ctx.tasks))
    assert len(seen) == 1
    applied = seen[0]["applied"]
    assert [name for name, _ in applied] == \
        [f"ctrl_{t.name}" for t in ctx.tasks], (
        "coherence probe must see the sequential composed state, "
        f"got {applied}")
    assert all(sign == 1.0 for _, sign in applied)
    assert "coherence_base" in res


def test_naive_stack_probes_composed_state(ctx, monkeypatch):
    seen = []
    _spy_probe(ctx, monkeypatch, seen)
    res = run_naive_stack(ctx, list(ctx.tasks))
    assert len(seen) == 1
    assert all(n == len(ctx.tasks) for n in seen[0]["lora_enabled"]), (
        "coherence probe must see all LoRA modules re-enabled after the "
        f"reversibility test, got per-site counts {seen[0]['lora_enabled']}")
    assert "coherence_base" in res


def test_finalize_resets_after_probe(ctx):
    run_coeff_addition(ctx, list(ctx.tasks))
    assert ctx.bank.applied == [], "_finalize must leave adapters reset"

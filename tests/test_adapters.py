import torch

from sequential_adapt.adapters import LoRAAdapter
from sequential_adapt.model import batch_forward_logits, params_unchanged
from sequential_adapt.train import fit_lora_adapter, fit_task_coefficients

PROBE = ["In domain A, the word blicket maps to the color"]


def _logits(ctx):
    return batch_forward_logits(ctx.model, ctx.tokenizer, PROBE, ctx.cfg.device)


def test_no_adapter_is_noop(ctx):
    ctx.reset_adapters()
    a = _logits(ctx)
    b = _logits(ctx)
    assert torch.equal(a, b)


def test_shared_adapter_changes_output(ctx):
    ctx.reset_adapters()
    base = _logits(ctx)
    ctx.bank.new_task("probe_task", requires_grad=False)
    with torch.no_grad():
        ctx.bank.tasks["probe_task"]["coeffs"].fill_(1.0)
    ctx.bank.apply([("probe_task", 1.0)])
    adapted = _logits(ctx)
    assert not torch.allclose(base, adapted)
    ctx.reset_adapters()


def test_enable_disable_restores_base(ctx):
    ctx.reset_adapters()
    base = _logits(ctx)
    ctx.bank.new_task("toggle_task", requires_grad=False)
    with torch.no_grad():
        ctx.bank.tasks["toggle_task"]["coeffs"].fill_(0.5)
    ctx.bank.apply([("toggle_task", 1.0)])
    assert not torch.allclose(base, _logits(ctx))
    ctx.bank.clear()
    assert torch.equal(base, _logits(ctx))


def test_negation_exactly_reverses(ctx):
    """Coefficient negation must cancel in parameter space -> identical logits."""
    ctx.reset_adapters()
    base = _logits(ctx)
    ctx.bank.new_task("rev_task", requires_grad=False)
    with torch.no_grad():
        ctx.bank.tasks["rev_task"]["coeffs"].normal_()
    ctx.bank.apply([("rev_task", 1.0), ("rev_task", -1.0)])
    assert torch.allclose(base, _logits(ctx), atol=1e-5)
    ctx.reset_adapters()


def test_base_frozen_after_coefficient_fit(ctx):
    ctx.reset_adapters()
    task = ctx.tasks[0]
    fit_task_coefficients(ctx.model, ctx.tokenizer, ctx.bank, task, ctx.cfg)
    assert params_unchanged(ctx.model, ctx.param_snapshot)
    ctx.reset_adapters()


def test_base_frozen_after_lora_fit(ctx):
    ctx.reset_adapters()
    task = ctx.tasks[0]
    name = fit_lora_adapter(ctx.model, ctx.tokenizer, ctx.sites, task, ctx.cfg,
                            adapter_name="lora_freeze_test")
    assert params_unchanged(ctx.model, ctx.param_snapshot)
    for site in ctx.sites.values():
        site.set_enabled(name, False)
    ctx.reset_adapters()


def test_lora_zero_init_is_noop():
    ad = LoRAAdapter(8, 8, rank=2, alpha=4)
    x = torch.randn(3, 8)
    assert torch.equal(ad(x), torch.zeros(3, 8))


def test_fitting_reduces_loss(ctx):
    ctx.reset_adapters()
    task = ctx.tasks[0]
    losses = fit_task_coefficients(ctx.model, ctx.tokenizer, ctx.bank, task,
                                   ctx.cfg)
    assert losses[-1] < losses[0]
    ctx.reset_adapters()


def test_fitting_with_drift_anchor_runs(ctx):
    import dataclasses
    ctx.reset_adapters()
    cfg = dataclasses.replace(ctx.cfg, anchor_weight=0.5, steps=5)
    task = ctx.tasks[0]
    losses = fit_task_coefficients(ctx.model, ctx.tokenizer, ctx.bank, task, cfg)
    assert len(losses) == 5 and all(l == l for l in losses)  # no NaN
    ctx.reset_adapters()

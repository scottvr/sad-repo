"""Tests for composition-aware replay and hard orthogonal projection."""
import dataclasses

from sequential_adapt.train import _replay_pairs, fit_task_coefficients


def test_replay_fraction_subsamples_deterministically(ctx):
    cfg = ctx.cfg
    t0 = ctx.tasks[0]
    full = _replay_pairs([t0], dataclasses.replace(cfg, replay_fraction=1.0),
                         "task_B")
    half_cfg = dataclasses.replace(cfg, replay_fraction=0.5)
    half = _replay_pairs([t0], half_cfg, "task_B")
    assert len(half) == max(1, round(len(full) * 0.5))
    assert set(half) <= set(full)
    assert _replay_pairs([t0], half_cfg, "task_B") == half  # deterministic
    # Every earlier task keeps at least one example, even at tiny fractions.
    tiny = _replay_pairs(list(ctx.tasks), dataclasses.replace(
        cfg, replay_fraction=0.01), "task_Z")
    assert len(tiny) == len(ctx.tasks)


def test_replay_loss_runs_and_is_finite(ctx):
    ctx.reset_adapters()
    cfg = dataclasses.replace(ctx.cfg, replay_weight=1.0, steps=5)
    t0, t1 = ctx.tasks[0], ctx.tasks[1]
    fit_task_coefficients(ctx.model, ctx.tokenizer, ctx.bank, t0, cfg)
    losses = fit_task_coefficients(
        ctx.model, ctx.tokenizer, ctx.bank, t1, cfg,
        prev_applied=[(t0.name, 1.0)], prev_task_names=[t0.name],
        replay_tasks=[t0])
    assert len(losses) == 5 and all(l == l for l in losses)  # no NaN
    ctx.reset_adapters()


def test_replay_increases_loss_vs_no_replay(ctx):
    """With identical seeds/steps, adding the replay CE term must change
    (and at step 0, strictly increase) the reported loss."""
    ctx.reset_adapters()
    t0, t1 = ctx.tasks[0], ctx.tasks[1]
    base_cfg = dataclasses.replace(ctx.cfg, steps=3)
    fit_task_coefficients(ctx.model, ctx.tokenizer, ctx.bank, t0, base_cfg)

    without = fit_task_coefficients(
        ctx.model, ctx.tokenizer, ctx.bank, t1, base_cfg,
        prev_applied=[(t0.name, 1.0)], prev_task_names=[t0.name])
    with_replay = fit_task_coefficients(
        ctx.model, ctx.tokenizer, ctx.bank, t1,
        dataclasses.replace(base_cfg, replay_weight=1.0),
        prev_applied=[(t0.name, 1.0)], prev_task_names=[t0.name],
        replay_tasks=[t0])
    # Coefficients start at zero both times, so step-0 losses differ exactly
    # by the (positive) replay CE term.
    assert with_replay[0] > without[0]
    ctx.reset_adapters()


def test_hard_ortho_projects_exactly(ctx):
    """After fitting with hard_ortho (gates off), the new task's raw
    coefficients must be orthogonal to the earlier task's direction."""
    ctx.reset_adapters()
    cfg = dataclasses.replace(ctx.cfg, hard_ortho=True, steps=10)
    t0, t1 = ctx.tasks[0], ctx.tasks[1]
    fit_task_coefficients(ctx.model, ctx.tokenizer, ctx.bank, t0, cfg)
    fit_task_coefficients(
        ctx.model, ctx.tokenizer, ctx.bank, t1, cfg,
        prev_applied=[(t0.name, 1.0)], prev_task_names=[t0.name])
    d0 = ctx.bank.flat_coeffs(t0.name)
    d0 = d0 / d0.norm()
    c1 = ctx.bank.tasks[t1.name]["coeffs"].detach().reshape(-1)
    assert c1.norm() > 0  # fitting actually moved the coefficients
    cos = float((c1 @ d0) / (c1.norm() + 1e-8))
    assert abs(cos) < 1e-4
    ctx.reset_adapters()


def test_hard_ortho_with_replay_runs(ctx):
    ctx.reset_adapters()
    cfg = dataclasses.replace(ctx.cfg, hard_ortho=True, replay_weight=0.5,
                              steps=5)
    t0, t1 = ctx.tasks[0], ctx.tasks[1]
    fit_task_coefficients(ctx.model, ctx.tokenizer, ctx.bank, t0, cfg)
    losses = fit_task_coefficients(
        ctx.model, ctx.tokenizer, ctx.bank, t1, cfg,
        prev_applied=[(t0.name, 1.0)], prev_task_names=[t0.name],
        replay_tasks=[t0])
    assert len(losses) == 5 and all(l == l for l in losses)
    ctx.reset_adapters()

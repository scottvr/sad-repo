"""Experiment orchestration: baselines, controller method, full suite.

Methods compared (all on the same frozen base + same synthetic tasks):

  independent     one shared-basis coefficient vector fitted per task, alone.
  naive_stack     per-task LoRA modules trained sequentially, each on top of
                  the previous ones, no control (baseline 2).
  coeff_add       task-vector arithmetic: sum the independently fitted
                  coefficient vectors (baseline 3).
  controller      sequential coefficient fitting with previous tasks active,
                  interference (cosine) penalty, per-site gates (Model C),
                  plus an MLP controller (Model B) that maps frozen-base
                  context embeddings to coefficients for routed evaluation.

Every method reports stage-wise task evals, forgetting, drift on neutral
probes, coherence probes, and a reversibility test (remove the first task's
update after the whole sequence).
"""

import json
import os
import time

import torch

from .adapters import AdapterBank, attach_adapter_sites
from .controllers import LookupController, MLPController, train_mlp_controller
from .data import context_prompts, make_tasks
from .eval import (coherence_probe, drift_kl, eval_all_tasks, neutral_logits)
from .metrics import reversibility, summarize_sequence
from .model import (assert_frozen, context_embedding, load_frozen_model,
                    params_unchanged, snapshot_params)
from .train import fit_lora_adapter, fit_task_coefficients


class ExperimentContext:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log or (lambda *_: None)
        torch.manual_seed(cfg.seed)
        self.model, self.tokenizer = load_frozen_model(cfg.model_name, cfg.device)
        assert_frozen(self.model)
        self.param_snapshot = snapshot_params(self.model)
        self.sites = attach_adapter_sites(self.model, cfg.site_suffixes)
        self.bank = AdapterBank(self.sites, cfg.rank, cfg.n_components,
                                cfg.seed, device=cfg.device)
        self.tasks = make_tasks(cfg.n_tasks, cfg.facts_per_task, cfg.label_space)
        self.task_by_name = {t.name: t for t in self.tasks}
        self.reset_adapters()
        self.base_neutral = neutral_logits(self.model, self.tokenizer, cfg)
        self.base_eval = eval_all_tasks(self.model, self.tokenizer, self.tasks, cfg)

    def reset_adapters(self):
        """Disable every adapter: model behaves as the frozen base."""
        self.bank.clear()
        for site in self.sites.values():
            for name in site.adapters:
                site.set_enabled(name, name == "shared")  # shared is no-op when cleared

    def check_frozen(self):
        if not params_unchanged(self.model, self.param_snapshot):
            raise AssertionError("Base model weights changed!")

    def stage_snapshot(self, stage_name):
        evals = eval_all_tasks(self.model, self.tokenizer, self.tasks, self.cfg)
        drift = drift_kl(self.base_neutral,
                         neutral_logits(self.model, self.tokenizer, self.cfg))
        return {"stage": stage_name, "task_evals": evals, "drift_kl": drift}


def _finalize(ctx, stages, order_names, extra=None):
    out = summarize_sequence(stages, order_names, base_eval=ctx.base_eval)
    out["stages"] = stages
    out["final_drift_kl"] = stages[-1]["drift_kl"]
    out["coherence"] = coherence_probe(ctx.model, ctx.tokenizer, ctx.tasks, ctx.cfg)
    if extra:
        out.update(extra)
    return out


def run_independent(ctx):
    """Baseline 1: fit each task alone from base; also caches fits for reuse."""
    results = {}
    for task in ctx.tasks:
        name = f"indep_{task.name}"
        if name not in ctx.bank.tasks:
            ctx.reset_adapters()
            fit_task_coefficients(ctx.model, ctx.tokenizer, ctx.bank, task,
                                  ctx.cfg, log=ctx.log)
            ctx.bank.set_task(name, ctx.bank.tasks[task.name]["coeffs"])
        ctx.bank.apply([(name, 1.0)])
        evals = eval_all_tasks(ctx.model, ctx.tokenizer, ctx.tasks, ctx.cfg)
        drift = drift_kl(ctx.base_neutral,
                         neutral_logits(ctx.model, ctx.tokenizer, ctx.cfg))
        results[task.name] = {"task_evals": evals, "drift_kl": drift}
    ctx.reset_adapters()
    ctx.check_frozen()
    return {"per_task": results, "base_evals": ctx.base_eval}


def run_naive_stack(ctx, order):
    """Baseline 2: LoRA modules trained sequentially, stacked, no control."""
    ctx.reset_adapters()
    order_names = [t.name for t in order]
    stages, adapter_names = [], []
    for task in order:
        aname = fit_lora_adapter(ctx.model, ctx.tokenizer, ctx.sites, task,
                                 ctx.cfg, log=ctx.log)
        adapter_names.append(aname)
        stages.append(ctx.stage_snapshot(f"after_{task.name}"))
    # Reversibility: disable the FIRST task's module, keep the rest.
    before = stages[-1]["task_evals"]
    for site in ctx.sites.values():
        site.set_enabled(adapter_names[0], False)
    after = eval_all_tasks(ctx.model, ctx.tokenizer, ctx.tasks, ctx.cfg)
    rev = reversibility(ctx.base_eval, before, after,
                        order_names[0], order_names[1:])
    rev["after_reversal_evals"] = after
    ctx.reset_adapters()
    ctx.check_frozen()
    return _finalize(ctx, stages, order_names, {"reversibility": rev})


def run_coeff_addition(ctx, order):
    """Baseline 3: sum independently fitted coefficient vectors (task arithmetic)."""
    run_independent(ctx)  # ensures indep_* fits exist (cached)
    order_names = [t.name for t in order]
    stages = []
    applied = []
    for task in order:
        applied.append((f"indep_{task.name}", 1.0))
        ctx.bank.apply(list(applied))
        stages.append(ctx.stage_snapshot(f"after_{task.name}"))
    before = stages[-1]["task_evals"]
    # Reversibility: negation is exact in parameter space; test behavior.
    ctx.bank.apply(list(applied) + [(f"indep_{order_names[0]}", -1.0)])
    after = eval_all_tasks(ctx.model, ctx.tokenizer, ctx.tasks, ctx.cfg)
    rev = reversibility(ctx.base_eval, before, after,
                        order_names[0], order_names[1:])
    rev["after_reversal_evals"] = after
    ctx.reset_adapters()
    ctx.check_frozen()
    return _finalize(ctx, stages, order_names, {"reversibility": rev})


def run_controller(ctx, order):
    """Proposed method: sequential coefficient fitting with interference
    penalty and gates, plus MLP controller for routed evaluation."""
    cfg = ctx.cfg
    ctx.reset_adapters()
    order_names = [t.name for t in order]
    stages, applied, fitted, done_tasks = [], [], [], []
    for task in order:
        fit_task_coefficients(
            ctx.model, ctx.tokenizer, ctx.bank, task, cfg,
            prev_applied=list(applied), prev_task_names=list(fitted),
            replay_tasks=list(done_tasks),
            train_gates=cfg.train_gates, log=ctx.log)
        cname = f"ctrl_{task.name}"
        ctx.bank.set_task(cname, ctx.bank.tasks[task.name]["coeffs"],
                          ctx.bank.tasks[task.name]["gate_logits"])
        applied.append((cname, 1.0))
        fitted.append(cname)
        done_tasks.append(task)
        ctx.bank.apply(list(applied))
        stages.append(ctx.stage_snapshot(f"after_{task.name}"))

    before = stages[-1]["task_evals"]
    ctx.bank.apply(list(applied) + [(f"ctrl_{order_names[0]}", -1.0)])
    after = eval_all_tasks(ctx.model, ctx.tokenizer, ctx.tasks, ctx.cfg)
    rev = reversibility(ctx.base_eval, before, after,
                        order_names[0], order_names[1:])
    rev["after_reversal_evals"] = after

    # --- Cramming diagnostic: each sequentially fitted vector applied ALONE.
    # Under replay, "perfect composed retention" could mean the newest vector
    # simply re-learned every earlier task (it has capacity for the whole
    # corpus) rather than repairing the composition. If the newest vector
    # alone solves earlier tasks' probes, it's cramming, not repair.
    solo = {}
    for cname, tname in zip(fitted, order_names):
        ctx.bank.apply([(cname, 1.0)])
        solo[tname] = eval_all_tasks(ctx.model, ctx.tokenizer, ctx.tasks,
                                     ctx.cfg)
    earlier = order_names[:-1]
    newest_alone_on_earlier = (
        sum(solo[order_names[-1]][t]["acc"] for t in earlier) / len(earlier)
        if earlier else None)

    # --- Model B: controller mapping context embeddings -> coefficients ---
    # Targets are the INDEPENDENTLY fitted coefficients: sequentially fitted
    # ones are only valid with earlier tasks applied, so they are the wrong
    # thing to route to in isolation. Routing asks: can a function of
    # frozen-base context embeddings select a standalone-valid update?
    run_independent(ctx)  # ensures indep_* fits are cached
    ctx.reset_adapters()  # context embeddings come from the frozen base
    lookup = LookupController()
    ctx_embs, targets, heldout = [], [], []
    for task in order:
        flat = ctx.bank.flat_coeffs(f"indep_{task.name}")
        lookup.add(task.name, flat)
        prompts = context_prompts(task)
        for p in prompts[:-1]:  # last phrasing held out
            ctx_embs.append(context_embedding(ctx.model, ctx.tokenizer, [p],
                                              cfg.device))
            targets.append(flat)
        heldout.append((task.name,
                        context_embedding(ctx.model, ctx.tokenizer,
                                          [prompts[-1]], cfg.device), flat))
    X = torch.stack(ctx_embs)
    Y = torch.stack(targets)
    mlp = MLPController(X.shape[1], Y.shape[1],
                        cfg.controller_hidden).to(cfg.device)
    losses = train_mlp_controller(mlp, X, Y, cfg.controller_steps,
                                  cfg.controller_lr)
    heldout_mse = float(torch.stack([
        torch.nn.functional.mse_loss(mlp.predict(context_emb=e), f)
        for _, e, f in heldout]).mean())

    # Routed eval: per task, apply the controller's predicted coefficients
    # (from a held-out context phrasing) and measure that task's accuracy.
    routed = {}
    for tname, emb, _ in heldout:
        ctx.bank.apply_flat(mlp.predict(context_emb=emb))
        routed[tname] = eval_all_tasks(ctx.model, ctx.tokenizer, ctx.tasks,
                                       ctx.cfg)[tname]
    ctx.reset_adapters()
    ctx.check_frozen()
    extra = {
        "reversibility": rev,
        "solo_evals": solo,
        "newest_alone_on_earlier": newest_alone_on_earlier,
        "gates": {n: ctx.bank.gates(f"ctrl_{n}").tolist() for n in order_names},
        "controller": {
            "final_train_mse": losses[-1],
            "heldout_context_mse": heldout_mse,
            "routed_evals": routed,
        },
    }
    return _finalize(ctx, stages, order_names, extra)


def run_full_suite(cfg, methods=("independent", "naive_stack", "coeff_add",
                                 "controller"),
                   order_sensitivity_check=True, log=print):
    """Run everything; returns one JSON-serializable dict."""
    t0 = time.time()
    ctx = ExperimentContext(cfg, log=log)
    fwd = list(ctx.tasks)
    rev_order = list(reversed(ctx.tasks))
    results = {"config": cfg.to_dict(), "base_evals": ctx.base_eval,
               "methods": {}}

    runners = {"naive_stack": run_naive_stack,
               "coeff_add": run_coeff_addition,
               "controller": run_controller}
    for m in methods:
        log(f"== {m} ==")
        if m == "independent":
            results["methods"][m] = run_independent(ctx)
            continue
        entry = {"forward": runners[m](ctx, fwd)}
        if order_sensitivity_check:
            log(f"== {m} (reversed order) ==")
            entry["reversed"] = runners[m](ctx, rev_order)
            from .metrics import order_sensitivity
            entry["order_sensitivity"] = order_sensitivity(
                entry["forward"]["final_evals"],
                entry["reversed"]["final_evals"])
        results["methods"][m] = entry
    results["runtime_sec"] = round(time.time() - t0, 1)
    return results


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def format_table(results):
    """Concise per-method summary table as a string."""
    lines = []
    header = (f"{'method':<14}{'final acc (per task)':<26}{'avg ret':<9}"
              f"{'forget':<8}{'drift':<8}{'rev gap':<9}{'collat':<8}"
              f"{'ordsens':<8}")
    lines.append(header)
    lines.append("-" * len(header))
    base = results["base_evals"]
    base_accs = " ".join(f"{v['acc']:.2f}" for v in base.values())
    lines.append(f"{'base':<14}{base_accs:<26}{'-':<9}{'-':<8}{'0.000':<8}"
                 f"{'-':<9}{'-':<8}{'-':<8}")
    for m, entry in results["methods"].items():
        if m == "independent":
            accs = " ".join(
                f"{entry['per_task'][t]['task_evals'][t]['acc']:.2f}"
                for t in entry["per_task"])
            drift = max(v["drift_kl"] for v in entry["per_task"].values())
            lines.append(f"{'independent':<14}{accs:<26}{'-':<9}{'-':<8}"
                         f"{drift:<8.3f}{'-':<9}{'-':<8}{'-':<8}")
            continue
        fwd = entry["forward"]
        accs = " ".join(f"{v['acc']:.2f}" for v in fwd["final_evals"].values())
        avg_ret = fwd["average_retention"]
        forget = sum(fwd["forgetting"].values()) / max(len(fwd["forgetting"]), 1)
        rev = fwd["reversibility"]
        osens = entry.get("order_sensitivity", {}).get("mean_abs_acc_diff")
        lines.append(
            f"{m:<14}{accs:<26}"
            f"{avg_ret if avg_ret is not None else float('nan'):<9.2f}"
            f"{forget:<8.2f}{fwd['final_drift_kl']:<8.3f}"
            f"{rev['return_to_base_gap']:<9.2f}{rev['collateral']:<8.2f}"
            f"{osens if osens is not None else float('nan'):<8.2f}")
        if m == "controller":
            routed = fwd["controller"]["routed_evals"]
            raccs = " ".join(f"{v['acc']:.2f}" for v in routed.values())
            lines.append(f"{'  (routed)':<14}{raccs:<26}"
                         f"(controller-predicted coeffs, held-out contexts)")
    return "\n".join(lines)

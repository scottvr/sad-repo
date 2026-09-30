"""Final session: template pools, prompting baseline, held-out-pool
recording, and the pre-registered decision rule."""
import importlib.util
import os

import pytest

from sequential_adapt.config import Config
from sequential_adapt.data import (HELDOUT_TEMPLATE_POOL, TEMPLATES,
                                   TRAIN_TEMPLATE_ORDER, icl_context,
                                   make_tasks, train_templates)

LABELS = (" red", " blue", " green", " yellow", " purple", " orange")


def _load_summarizer():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts",
                        "summarize_final.py")
    spec = importlib.util.spec_from_file_location("summarize_final", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- templates ------------------------------------------------------------

def test_historical_templates_unmoved():
    assert TEMPLATES[0] == "In domain {d}, the word {w} maps to the color"
    assert TEMPLATES[1] == "Domain {d} rule: {w} is assigned the color"
    assert TEMPLATES[2] == ("Within domain {d}, the color associated with "
                            "{w} is")
    assert train_templates(2) == (0, 1) == Config(device="cpu").train_templates


def test_pools_disjoint_and_complete():
    assert not set(TRAIN_TEMPLATE_ORDER) & set(HELDOUT_TEMPLATE_POOL)
    assert set(TRAIN_TEMPLATE_ORDER) | set(HELDOUT_TEMPLATE_POOL) == \
        set(range(len(TEMPLATES)))
    assert Config(device="cpu").heldout_pool == HELDOUT_TEMPLATE_POOL
    for t in TEMPLATES:
        assert t.format(d="A", w="dax")  # well-formed


def test_trained_heldout_overlap_rejected():
    with pytest.raises(ValueError):
        Config(device="cpu", train_templates=(0, 2))


def test_icl_context_states_every_fact_once():
    tasks = make_tasks(3, 4, LABELS)
    facts = [f for t in tasks for f in t.facts]
    ctx = icl_context(facts, seed=1)
    for f in facts:
        assert ctx.count(f"In domain {f.domain}, {f.word} is{f.label}.") == 1
    assert icl_context(facts, seed=1) == ctx          # deterministic
    assert icl_context(facts, seed=2) != ctx          # seed shuffles order


# --- torch-backed: prompting baseline + recorded pool metrics ---------------

def test_icl_suite_shapes(ctx):
    from sequential_adapt.icl import CONDITIONS, run_icl_suite
    res = run_icl_suite(ctx.model, ctx.tokenizer, ctx.tasks, ctx.cfg)
    assert set(res) == set(CONDITIONS)
    for r in res.values():
        assert 0.0 <= r["heldout_pool_acc"] <= 1.0
        for per_task in r["heldout_pool"].values():
            assert set(per_task["per_template"]) == set(HELDOUT_TEMPLATE_POOL)


def test_controller_records_heldout_pool(ctx):
    from sequential_adapt.experiments import run_controller
    out = run_controller(ctx, list(ctx.tasks))
    pool = out["final_evals_heldout_pool"]
    assert set(pool) == {t.name for t in ctx.tasks}
    for routed in out["controller"]["routed_evals"].values():
        assert set(routed["heldout_pool"]["per_template"]) == \
            set(HELDOUT_TEMPLATE_POOL)


# --- decision rule (pure) --------------------------------------------------

def _agg(**means):
    return {"n_seeds": 5, **{k: {"mean": v, "std": 0.0}
                             for k, v in means.items()}}


def _world(adapter_comp, adapter_route, p_same, p_large, retention=1.0):
    """All six arms identical; prompting identical across conditions."""
    sf = _load_summarizer()
    agg = {("coef", a): _agg(composed_heldout=adapter_comp,
                             routed_heldout=adapter_route,
                             retention=retention)
           for a in sf.COEF_ARMS}
    agg[("icl", "distilgpt2_default")] = _agg(icl_all_heldout=p_same,
                                              icl_domain_heldout=p_same)
    agg[("icl", "gpt2-medium_default")] = _agg(icl_all_heldout=p_large,
                                               icl_domain_heldout=p_large)
    return sf.decide(agg)


@pytest.mark.parametrize("args,verdict,reason", [
    ((0.9, 0.9, 0.3, 0.4), "CONTINUE", "winning"),
    ((0.2, 0.2, 0.9, 0.9), "STOP", "prompting wins"),
    ((0.8, 0.8, 0.5, 0.75), "STOP", "edge vanishes"),
    ((0.2, 0.25, 0.1, 0.1), "STOP", "no transfer"),
    ((0.5, 0.5, 0.3, 0.3), "STOP", "inconclusive"),
])
def test_decision_rule(args, verdict, reason):
    d = _world(*args)
    assert d["verdict"] == verdict
    assert reason in d["reason"]


def test_composed_needs_retention():
    # Great composed held-out but broken retention does not count; routed
    # pairing alone is below the bar, so this must not CONTINUE.
    d = _world(0.95, 0.4, 0.2, 0.2, retention=0.5)
    assert d["verdict"] == "STOP"
    assert d["pairings"]["composed_vs_icl_all"]["arm"] is None


def test_decision_incomplete_without_prompting():
    sf = _load_summarizer()
    agg = {("coef", "d2_k8r4"): _agg(composed_heldout=0.9,
                                     routed_heldout=0.9, retention=1.0)}
    assert sf.decide(agg)["verdict"] == "INCOMPLETE"

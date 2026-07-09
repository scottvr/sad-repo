import math

from sequential_adapt.metrics import (average_retention, forgetting,
                                      order_sensitivity, retention_matrix,
                                      reversibility, summarize_sequence)


def _stage(name, accs):
    return {"stage": name,
            "task_evals": {t: {"acc": a, "loss": 1 - a} for t, a in accs.items()}}


ORDER = ["task_A", "task_B"]
STAGES = [
    _stage("after_A", {"task_A": 1.0, "task_B": 0.2}),
    _stage("after_B", {"task_A": 0.6, "task_B": 0.9}),
]


def test_retention_matrix():
    R = retention_matrix(STAGES, ORDER)
    assert R == [[1.0, 0.2], [0.6, 0.9]]


def test_forgetting_sane():
    f = forgetting(STAGES, ORDER)
    assert math.isclose(f["task_A"], 0.4, abs_tol=1e-9)  # 1.0 -> 0.6
    assert math.isclose(f["task_B"], 0.0, abs_tol=1e-9)  # learned last
    assert all(-1.0 <= v <= 1.0 for v in f.values())


def test_average_retention():
    assert math.isclose(average_retention(STAGES, ORDER), 0.6, abs_tol=1e-9)


def test_order_sensitivity():
    fwd = {"task_A": {"acc": 0.6}, "task_B": {"acc": 0.9}}
    rev = {"task_A": {"acc": 0.9}, "task_B": {"acc": 0.5}}
    o = order_sensitivity(fwd, rev)
    assert math.isclose(o["mean_abs_acc_diff"], (0.3 + 0.4) / 2, abs_tol=1e-9)
    assert math.isclose(o["per_task"]["task_A"], -0.3, abs_tol=1e-9)


def test_reversibility_perfect():
    base = {"task_A": {"acc": 0.1}, "task_B": {"acc": 0.2}}
    before = {"task_A": {"acc": 0.9}, "task_B": {"acc": 0.9}}
    after = {"task_A": {"acc": 0.1}, "task_B": {"acc": 0.9}}
    r = reversibility(base, before, after, "task_A", ["task_B"])
    assert r["return_to_base_gap"] == 0.0
    assert r["collateral"] == 0.0


def test_reversibility_collateral():
    base = {"task_A": {"acc": 0.1}, "task_B": {"acc": 0.2}}
    before = {"task_A": {"acc": 0.9}, "task_B": {"acc": 0.9}}
    after = {"task_A": {"acc": 0.5}, "task_B": {"acc": 0.4}}
    r = reversibility(base, before, after, "task_A", ["task_B"])
    assert math.isclose(r["return_to_base_gap"], 0.4, abs_tol=1e-9)
    assert math.isclose(r["collateral"], 0.5, abs_tol=1e-9)


def test_summarize_sequence_keys():
    s = summarize_sequence(STAGES, ORDER)
    for key in ("task_order", "retention_matrix", "forgetting",
                "average_retention", "final_evals"):
        assert key in s

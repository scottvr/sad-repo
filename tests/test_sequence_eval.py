import json

from sequential_adapt.experiments import (run_coeff_addition, run_controller,
                                          run_full_suite, save_results)


def _check_sequence_result(res, n_tasks):
    for key in ("task_order", "retention_matrix", "forgetting",
                "average_retention", "final_evals", "stages",
                "final_drift_kl", "coherence", "reversibility"):
        assert key in res, f"missing key {key}"
    assert len(res["retention_matrix"]) == n_tasks
    for row in res["retention_matrix"]:
        assert len(row) == n_tasks
        assert all(0.0 <= v <= 1.0 for v in row)
    assert res["final_drift_kl"] >= -1e-6
    rev = res["reversibility"]
    assert 0.0 <= rev["return_to_base_gap"] <= 1.0
    assert 0.0 <= rev["collateral"] <= 1.0
    coh = res["coherence"]
    assert 0.0 <= coh["paraphrase_consistency"] <= 1.0
    assert 0.0 <= coh["offdomain_leakage"] <= 1.0


def test_coeff_addition_sequence(ctx):
    res = run_coeff_addition(ctx, list(ctx.tasks))
    _check_sequence_result(res, len(ctx.tasks))


def test_controller_sequence(ctx):
    res = run_controller(ctx, list(ctx.tasks))
    _check_sequence_result(res, len(ctx.tasks))
    c = res["controller"]
    assert "routed_evals" in c and len(c["routed_evals"]) == len(ctx.tasks)
    assert c["final_train_mse"] >= 0.0
    assert "gates" in res


def test_full_suite_serializable(cfg, tmp_path):
    res = run_full_suite(cfg, methods=("independent", "coeff_add"),
                         order_sensitivity_check=False, log=lambda *_: None)
    path = save_results(res, str(tmp_path / "res.json"))
    with open(path) as f:
        loaded = json.load(f)
    assert "methods" in loaded and "config" in loaded
    assert "coeff_add" in loaded["methods"]

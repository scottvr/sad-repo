import json
import importlib.util
import sys
from pathlib import Path


def _load_script_module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _eval(acc):
    return {"acc": acc, "loss": 1.0 - acc}


def test_summary_aggregates_sequence_and_anchor_conditions(tmp_path):
    mod = _load_script_module("summarize_multiseed")
    sequence_path = tmp_path / "sequence_seed_7.json"
    anchor_path = tmp_path / "controller_anchor_seed_7.json"
    sequence = {
        "config": {"seed": 7},
        "base_evals": {"task_A": _eval(0.25), "task_B": _eval(0.0)},
        "methods": {
            "independent": {
                "per_task": {
                    "task_A": {
                        "task_evals": {"task_A": _eval(1.0)},
                        "drift_kl": 2.0,
                    },
                    "task_B": {
                        "task_evals": {"task_B": _eval(0.75)},
                        "drift_kl": 3.0,
                    },
                }
            },
            "controller": {
                "forward": {
                    "task_order": ["task_A", "task_B"],
                    "final_evals": {
                        "task_A": _eval(0.5),
                        "task_B": _eval(1.0),
                    },
                    "average_retention": 0.5,
                    "final_drift_kl": 4.0,
                    "coherence": {"paraphrase_consistency": 0.25},
                    "reversibility": {
                        "return_to_base_gap": 0.0,
                        "collateral": 0.5,
                    },
                    "controller": {
                        "routed_evals": {
                            "task_A": _eval(1.0),
                            "task_B": _eval(1.0),
                        }
                    },
                },
                "order_sensitivity": {"mean_abs_acc_diff": 0.125},
            },
        },
    }
    anchor = {
        "config": {"seed": 7},
        "base_evals": sequence["base_evals"],
        "methods": {
            "controller": {
                "forward": {
                    "task_order": ["task_A", "task_B"],
                    "final_evals": {
                        "task_A": _eval(0.25),
                        "task_B": _eval(1.0),
                    },
                    "average_retention": 0.25,
                    "final_drift_kl": 0.4,
                    "coherence": {"paraphrase_consistency": 0.0},
                    "reversibility": {
                        "return_to_base_gap": 0.0,
                        "collateral": 0.25,
                    },
                    "controller": {
                        "routed_evals": {
                            "task_A": _eval(1.0),
                            "task_B": _eval(0.75),
                        }
                    },
                },
                "order_sensitivity": {"mean_abs_acc_diff": 0.25},
            },
        },
    }
    sequence_path.write_text(json.dumps(sequence))
    anchor_path.write_text(json.dumps(anchor))

    samples = mod.collect_samples([sequence_path], [anchor_path])
    summary, _ = mod.summarize(samples)

    assert summary["independent"]["final_acc_avg"]["mean"] == 0.875
    assert summary["independent"]["final_drift_kl"]["mean"] == 3.0
    assert summary["controller"]["average_retention"]["mean"] == 0.5
    assert summary["controller_routed"]["final_acc_avg"]["mean"] == 1.0
    assert summary["controller_anchor"]["final_drift_kl"]["mean"] == 0.4
    assert summary["controller_anchor_routed"]["final_acc_avg"]["mean"] == 0.875


def test_sweep_summary_uses_config_for_condition_labels(tmp_path):
    mod = _load_script_module("summarize_sweeps")
    path = tmp_path / "k" / "sequence_k_16_seed_2.json"
    path.parent.mkdir()
    data = {
        "config": {
            "seed": 2,
            "n_components": 16,
            "n_tasks": 2,
            "facts_per_task": 4,
            "ortho_penalty": 0.1,
            "anchor_weight": 0.0,
        },
        "base_evals": {"task_A": _eval(0.25), "task_B": _eval(0.0)},
        "methods": {
            "controller": {
                "forward": {
                    "task_order": ["task_A", "task_B"],
                    "final_evals": {
                        "task_A": _eval(0.5),
                        "task_B": _eval(1.0),
                    },
                    "average_retention": 0.5,
                    "final_drift_kl": 4.0,
                    "coherence": {"paraphrase_consistency": 0.0},
                    "reversibility": {
                        "return_to_base_gap": 0.0,
                        "collateral": 0.5,
                    },
                    "controller": {
                        "routed_evals": {
                            "task_A": _eval(1.0),
                            "task_B": _eval(1.0),
                        }
                    },
                },
                "order_sensitivity": {"mean_abs_acc_diff": 0.125},
            }
        },
    }
    path.write_text(json.dumps(data))

    samples = mod.collect_samples([path])
    summary, _ = mod.summarize(samples)

    assert summary["k=16|controller"]["final_acc_avg"]["mean"] == 0.75
    assert summary["k=16|controller_routed"]["final_acc_avg"]["mean"] == 1.0

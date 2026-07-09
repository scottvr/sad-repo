"""Pure metric computations over stage-wise evaluation records.

A "sequence record" is a list of stages; each stage is
    {"stage": str, "task_evals": {task_name: {"acc": float, "loss": float}}}
Stage i is the model state immediately after adapting to the i-th task.
"""


def retention_matrix(stages, task_order):
    """R[i][j] = accuracy on task_order[j] after stage i. Rows = stages."""
    return [[s["task_evals"][t]["acc"] for t in task_order] for s in stages]


def forgetting(stages, task_order):
    """Per task: (acc immediately after learning it) - (acc at final stage).

    Positive = forgot; ~0 = retained; negative = later tasks helped (backward
    transfer). Tasks learned at the final stage have forgetting 0 by def.
    """
    final = stages[-1]["task_evals"]
    out = {}
    for j, t in enumerate(task_order):
        if j >= len(stages):
            continue
        immediate = stages[j]["task_evals"][t]["acc"]
        out[t] = immediate - final[t]["acc"]
    return out


def average_retention(stages, task_order):
    """Mean over tasks of final accuracy on tasks learned before the last stage."""
    final = stages[-1]["task_evals"]
    learned_earlier = task_order[:max(len(stages) - 1, 0)]
    if not learned_earlier:
        return None
    return sum(final[t]["acc"] for t in learned_earlier) / len(learned_earlier)


def order_sensitivity(final_evals_fwd, final_evals_rev):
    """Mean |acc_fwd - acc_rev| per task over the two orderings' final states."""
    tasks = sorted(final_evals_fwd.keys())
    diffs = [abs(final_evals_fwd[t]["acc"] - final_evals_rev[t]["acc"])
             for t in tasks]
    return {
        "mean_abs_acc_diff": sum(diffs) / len(diffs),
        "per_task": {t: final_evals_fwd[t]["acc"] - final_evals_rev[t]["acc"]
                     for t in tasks},
    }


def reversibility(base_eval, before_reversal_eval, after_reversal_eval,
                  reversed_task, other_tasks):
    """Did removing one task's update return that task to base behavior while
    leaving the others alone?

    return_to_base_gap: |acc(reversed task after reversal) - acc at base|.
        0 = clean reversal of that task's behavior.
    collateral: mean |acc change| on other tasks caused by the reversal.
        0 = surgical removal.
    """
    gap = abs(after_reversal_eval[reversed_task]["acc"]
              - base_eval[reversed_task]["acc"])
    if other_tasks:
        collateral = sum(
            abs(after_reversal_eval[t]["acc"] - before_reversal_eval[t]["acc"])
            for t in other_tasks) / len(other_tasks)
    else:
        collateral = 0.0
    return {"return_to_base_gap": gap, "collateral": collateral}


def summarize_sequence(stages, task_order, base_eval=None):
    out = {
        "task_order": task_order,
        "retention_matrix": retention_matrix(stages, task_order),
        "forgetting": forgetting(stages, task_order),
        "average_retention": average_retention(stages, task_order),
        "final_evals": stages[-1]["task_evals"],
    }
    if base_eval is not None:
        out["base_evals"] = base_eval
    return out

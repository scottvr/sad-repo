"""In-context (prompting) baseline: no adapters, facts pasted into the prompt.

The question every adapter result has to answer: does it beat simply
telling the frozen model the facts? Scored with the same restricted argmax
over the label space as every adapter eval, on the same query phrasings.

Conditions (each the prompting analogue of an adapter condition):
  base        no facts in context            (= the frozen base floor)
  icl_domain  only the queried domain's facts (analogue of ROUTED adapters)
  icl_all     every domain's facts at once    (analogue of the COMPOSED state)
"""

import torch

from .data import (HELDOUT_TEMPLATE_POOL, TRAIN_TEMPLATE_ORDER,
                   check_single_token_labels, fact_prompt, icl_context)
from .model import batch_forward_logits

CONDITIONS = ("base", "icl_domain", "icl_all")


def icl_prompt(context: str, query: str) -> str:
    return f"{context}\n{query}" if context else query


@torch.no_grad()
def icl_accuracy(model, tokenizer, tasks, cfg, condition, templates, seed=0):
    """{task_name: {"acc": mean, "per_template": {idx: acc}}}."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    label_map = check_single_token_labels(tokenizer, cfg.label_space)
    label_ids = torch.tensor(list(label_map.values()))
    all_facts = [f for t in tasks for f in t.facts]
    out = {}
    for task in tasks:
        if condition == "base":
            context = ""
        elif condition == "icl_domain":
            context = icl_context(task.facts, seed)
        else:
            context = icl_context(all_facts, seed)
        per = {}
        for t_idx in templates:
            prompts = [icl_prompt(context, fact_prompt(f, t_idx))
                       for f in task.facts]
            gold = torch.tensor([label_map[f.label] for f in task.facts])
            logits = batch_forward_logits(model, tokenizer, prompts,
                                          cfg.device).cpu()
            pred = label_ids[logits[:, label_ids].argmax(dim=1)]
            per[int(t_idx)] = (pred == gold).float().mean().item()
        out[task.name] = {"acc": sum(per.values()) / len(per),
                          "per_template": per}
    return out


def run_icl_suite(model, tokenizer, tasks, cfg, seed=0):
    """Every condition on the training-phrasing pool and the held-out pool.

    For prompting nothing is trained, so the two pools differ only in
    phrasing; both are reported so adapter and prompting numbers line up
    column for column."""
    res = {}
    for cond in CONDITIONS:
        res[cond] = {
            "train_pool": icl_accuracy(model, tokenizer, tasks, cfg, cond,
                                       TRAIN_TEMPLATE_ORDER, seed),
            "heldout_pool": icl_accuracy(model, tokenizer, tasks, cfg, cond,
                                         HELDOUT_TEMPLATE_POOL, seed),
        }
        for pool in ("train_pool", "heldout_pool"):
            vals = [v["acc"] for v in res[cond][pool].values()]
            res[cond][f"{pool}_acc"] = sum(vals) / len(vals)
    return res

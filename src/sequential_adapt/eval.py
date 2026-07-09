"""Evaluation primitives: task accuracy/loss, drift, coherence probes."""

import torch
import torch.nn.functional as F

from .data import (NEUTRAL_PROBES, check_single_token_labels, fact_prompt,
                   task_prompts)
from .model import batch_forward_logits


@torch.no_grad()
def evaluate_task(model, tokenizer, task, cfg, template_idx=None):
    """Accuracy (argmax restricted to label space) and CE loss on one task.

    template_idx=None evaluates cfg.eval_template (a trained phrasing —
    retention of learned behavior); pass an int for a specific phrasing,
    e.g. cfg.heldout_template for generalization.
    """
    t_idx = cfg.eval_template if template_idx is None else template_idx
    pairs = task_prompts(task, t_idx)
    label_ids = list(check_single_token_labels(tokenizer, cfg.label_space).values())
    prompts = [p for p, _ in pairs]
    gold = torch.tensor([tokenizer.encode(a)[0] for _, a in pairs])
    logits = batch_forward_logits(model, tokenizer, prompts, cfg.device).cpu()
    loss = F.cross_entropy(logits, gold).item()
    restricted = logits[:, label_ids]
    pred = torch.tensor(label_ids)[restricted.argmax(dim=1)]
    acc = (pred == gold).float().mean().item()
    return {"acc": acc, "loss": loss}


@torch.no_grad()
def neutral_logits(model, tokenizer, cfg):
    return batch_forward_logits(model, tokenizer, NEUTRAL_PROBES, cfg.device)


@torch.no_grad()
def drift_kl(base_logits, adapted_logits):
    """Mean KL(base || adapted) over neutral probes, final position."""
    logp_base = F.log_softmax(base_logits, dim=-1)
    logp_adapt = F.log_softmax(adapted_logits, dim=-1)
    kl = (logp_base.exp() * (logp_base - logp_adapt)).sum(dim=-1)
    return kl.mean().item()


@torch.no_grad()
def coherence_probe(model, tokenizer, tasks, cfg):
    """Two probes of semantic coherence on the synthetic facts.

    paraphrase_consistency: for each fact, do two different phrasings get the
        same restricted-argmax label? (1.0 = coherent)
    offdomain_leakage: prompt a domain-A word inside domain-B's template; how
        often does the model still answer with A's label? (high = adaptations
        bleed across contexts instead of being context-conditioned)
    """
    label_ids = list(check_single_token_labels(tokenizer, cfg.label_space).values())
    label_id_t = torch.tensor(label_ids)

    def restricted_preds(prompts):
        logits = batch_forward_logits(model, tokenizer, prompts, cfg.device).cpu()
        return label_id_t[logits[:, label_ids].argmax(dim=1)]

    # Paraphrase consistency: template 0 vs held-out template.
    all_facts = [f for t in tasks for f in t.facts]
    p0 = restricted_preds([fact_prompt(f, 0) for f in all_facts])
    p1 = restricted_preds([fact_prompt(f, cfg.heldout_template) for f in all_facts])
    consistency = (p0 == p1).float().mean().item()

    # Off-domain leakage: each fact asked under every OTHER domain's name.
    leak_prompts, leak_gold = [], []
    domains = [t.domain for t in tasks]
    for t in tasks:
        for f in t.facts:
            for d in domains:
                if d != f.domain:
                    fake = type(f)(domain=d, word=f.word, label=f.label)
                    leak_prompts.append(fact_prompt(fake, 0))
                    leak_gold.append(tokenizer.encode(f.label)[0])
    if leak_prompts:
        preds = restricted_preds(leak_prompts)
        leakage = (preds == torch.tensor(leak_gold)).float().mean().item()
    else:
        leakage = 0.0
    return {"paraphrase_consistency": consistency, "offdomain_leakage": leakage}


def eval_all_tasks(model, tokenizer, tasks, cfg):
    """{task_name: {acc, loss}} for every task (held-out template)."""
    return {t.name: evaluate_task(model, tokenizer, t, cfg) for t in tasks}

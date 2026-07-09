"""Frozen base model loading and context embeddings."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_frozen_model(model_name: str, device: str = "cpu"):
    """Load a causal LM, freeze every parameter, return (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()  # deterministic (no dropout); autograd still flows to adapter params
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def assert_frozen(model):
    bad = [n for n, p in model.named_parameters() if p.requires_grad]
    if bad:
        raise AssertionError(f"Base parameters not frozen: {bad[:5]}")


def snapshot_params(model):
    """Cheap fingerprint of base weights to verify they never change."""
    with torch.no_grad():
        return {n: p.detach().clone() for n, p in model.named_parameters()}


def params_unchanged(model, snapshot) -> bool:
    """True if every base parameter matches the snapshot.

    Robust to AdapterSite wrapping done after the snapshot: wrapped base
    params gain a `.base` path segment, and adapter params (under
    `.adapters.`) are not base weights and are skipped.
    """
    with torch.no_grad():
        seen = 0
        for n, p in model.named_parameters():
            if ".adapters." in n:
                continue
            key = n.replace(".base.", ".")
            if key not in snapshot:
                raise KeyError(f"Unexpected parameter {n!r}")
            seen += 1
            if not torch.equal(p, snapshot[key]):
                return False
        if seen != len(snapshot):
            raise AssertionError("Base parameter count changed")
    return True


@torch.no_grad()
def batch_forward_logits(model, tokenizer, prompts, device="cpu"):
    """Final-position logits for each prompt. Returns [len(prompts), vocab]."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    out = model(**enc)
    last_idx = enc["attention_mask"].sum(dim=1) - 1  # last non-pad position
    rows = torch.arange(len(prompts), device=out.logits.device)
    return out.logits[rows, last_idx, :]


@torch.no_grad()
def context_embedding(model, tokenizer, prompts, device="cpu"):
    """Mean over prompts of the mean last-hidden-state -> one vector [d_model].

    Used as the controller's input representation of "which domain am I in".
    Adapters should be disabled by the caller so this reflects the frozen base.
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[-1]  # [B, T, d]
    mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
    mean_per_prompt = (h * mask).sum(dim=1) / mask.sum(dim=1)
    return mean_per_prompt.mean(dim=0)

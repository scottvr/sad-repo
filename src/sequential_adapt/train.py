"""Fitting routines. Base weights stay frozen; only adapter tensors move."""

import zlib

import torch
import torch.nn.functional as F

from .data import NEUTRAL_PROBES, task_prompts


def build_batch(tokenizer, pairs, device="cpu"):
    """pairs: list of (prompt, answer_label). Returns (enc, gold_ids)."""
    prompts = [p for p, _ in pairs]
    gold = torch.tensor([tokenizer.encode(a)[0] for _, a in pairs], device=device)
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    return enc, gold


def final_logits(model, enc):
    """Final non-pad position logits, differentiable. [B, vocab]"""
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    last_idx = enc["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(enc["input_ids"].shape[0], device=enc["input_ids"].device)
    return out.logits[rows, last_idx, :]


def _training_pairs(task, cfg):
    pairs = []
    for t_idx in cfg.train_templates:
        pairs.extend(task_prompts(task, t_idx))
    return pairs


def fit_task_coefficients(model, tokenizer, bank, task, cfg,
                          prev_applied=(), prev_task_names=(),
                          train_gates=False, log=None):
    """Fit one task's shared-basis coefficients (and optionally gates).

    prev_applied: tasks kept active (frozen) during fitting — sequential mode.
    prev_task_names: tasks whose coefficient directions the new task is
        penalized for overlapping with (cosine^2 * cfg.ortho_penalty).
    cfg.anchor_weight > 0 adds a drift anchor: KL(pristine base || adapted)
        on neutral probes, discouraging off-task behavior change.
    """
    bank.new_task(task.name, train_gates=train_gates)

    anchor = None
    if cfg.anchor_weight > 0:
        prior_applied = list(bank.applied)
        bank.apply([])  # pristine base for the anchor target
        anchor_enc = tokenizer(list(NEUTRAL_PROBES), return_tensors="pt",
                               padding=True).to(cfg.device)
        with torch.no_grad():
            anchor_logp = F.log_softmax(final_logits(model, anchor_enc), dim=-1)
        anchor = (anchor_enc, anchor_logp)
        bank.apply(prior_applied)

    bank.apply(list(prev_applied) + [(task.name, 1.0)])
    params = bank.trainable_params(task.name)
    opt = torch.optim.Adam(params, lr=cfg.lr)
    enc, gold = build_batch(tokenizer, _training_pairs(task, cfg), cfg.device)

    prev_dirs = []
    for name in prev_task_names:
        d = bank.flat_coeffs(name)
        n = d.norm()
        if n > 1e-8:
            prev_dirs.append(d / n)

    losses = []
    for step in range(cfg.steps):
        opt.zero_grad()
        logits = final_logits(model, enc)
        loss = F.cross_entropy(logits, gold)
        coeffs = bank.tasks[task.name]["coeffs"]
        loss = loss + cfg.l2 * coeffs.pow(2).sum()
        if prev_dirs and cfg.ortho_penalty > 0:
            flat = coeffs.reshape(-1)
            norm = flat.norm() + 1e-8
            for d in prev_dirs:
                cos = (flat @ d) / norm
                loss = loss + cfg.ortho_penalty * cos ** 2
        if anchor is not None:
            anchor_enc, anchor_logp = anchor
            cur_logp = F.log_softmax(final_logits(model, anchor_enc), dim=-1)
            kl = (anchor_logp.exp() * (anchor_logp - cur_logp)).sum(-1).mean()
            loss = loss + cfg.anchor_weight * kl
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log and (step % 50 == 0 or step == cfg.steps - 1):
            log(f"  [{task.name}] step {step} loss {loss.item():.4f}")
    return losses


def fit_lora_adapter(model, tokenizer, sites, task, cfg, adapter_name=None,
                     log=None):
    """Train a standard per-task LoRA adapter (one module per site).

    Any already-enabled adapters at each site stay active during training
    (that is what makes naive stacking 'naive'). Returns the adapter name.
    """
    from .adapters import LoRAAdapter

    name = adapter_name or f"lora_{task.name}"
    gen = torch.Generator().manual_seed(cfg.seed + zlib.crc32(name.encode()) % 10000)
    params = []
    for site in sites.values():
        ad = LoRAAdapter(site.d_in, site.d_out, cfg.lora_rank, cfg.lora_alpha,
                         generator=gen).to(cfg.device)
        site.add_adapter(name, ad, enabled=True)
        params.extend(ad.parameters())

    opt = torch.optim.Adam(params, lr=cfg.lora_lr)
    enc, gold = build_batch(tokenizer, _training_pairs(task, cfg), cfg.device)
    for step in range(cfg.steps):
        opt.zero_grad()
        logits = final_logits(model, enc)
        loss = F.cross_entropy(logits, gold)
        loss.backward()
        opt.step()
        if log and (step % 50 == 0 or step == cfg.steps - 1):
            log(f"  [{name}] step {step} loss {loss.item():.4f}")
    for site in sites.values():
        for p in site.adapters[name].parameters():
            p.requires_grad_(False)
    return name

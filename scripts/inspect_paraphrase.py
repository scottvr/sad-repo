"""Dump actual per-fact predictions for every template under base /
composed / routed states — eyeball-level validation of the paraphrase
metric (written after the 2026-07-15 coherence-probe bug: trust nothing
aggregated until the raw outputs have been looked at once).

For each fact and each of the 3 templates it prints the restricted-argmax
label, the unrestricted top-5 tokens, and whether the prediction matches
gold, under three model states:
  base      — frozen model, no adapters
  composed  — all tasks' sequentially fitted vectors applied (replay-style)
  routed:T  — controller-predicted coefficients for the fact's own task

Usage (GPU box): python scripts/inspect_paraphrase.py --steps 200 --replay 1.0
Artifacts: JSON dump under artifacts/ (--out), table on stdout.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import torch

from sequential_adapt.config import Config
from sequential_adapt.data import (TEMPLATES, WIDE_LABEL_SPACE,
                                   check_single_token_labels, fact_prompt)
from sequential_adapt.eval import coherence_probe
from sequential_adapt.experiments import ExperimentContext, run_controller
from sequential_adapt.model import batch_forward_logits


@torch.no_grad()
def dump_state(ctx, state_name, out_rows):
    tok, cfg = ctx.tokenizer, ctx.cfg
    label_ids = list(check_single_token_labels(tok, cfg.label_space).values())
    label_id_t = torch.tensor(label_ids)
    for task in ctx.tasks:
        for f in task.facts:
            row = {"state": state_name, "task": task.name, "word": f.word,
                   "gold": f.label, "templates": {}}
            for t_idx in range(len(TEMPLATES)):
                logits = batch_forward_logits(
                    ctx.model, tok, [fact_prompt(f, t_idx)], cfg.device).cpu()[0]
                restricted = label_id_t[logits[label_ids].argmax()].item()
                top5 = logits.topk(5).indices.tolist()
                row["templates"][t_idx] = {
                    "restricted_pred": tok.decode([restricted]),
                    "top5": [tok.decode([i]) for i in top5],
                    "correct": tok.decode([restricted]) == f.label,
                }
            out_rows.append(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--n-components", "--k", dest="n_components", type=int,
                    default=8)
    ap.add_argument("--facts-per-task", type=int, default=4)
    ap.add_argument("--wide-labels", action="store_true")
    ap.add_argument("--replay", type=float, default=1.0,
                    help="replay weight for the composed state (default 1.0 "
                         "— the headline configuration)")
    ap.add_argument("--out", default="artifacts/paraphrase_inspection.json")
    args = ap.parse_args()
    label_kw = {"label_space": WIDE_LABEL_SPACE} if args.wide_labels else {}
    cfg = Config(model_name=args.model, steps=args.steps, seed=args.seed,
                 device=args.device, n_components=args.n_components,
                 facts_per_task=args.facts_per_task,
                 replay_weight=args.replay, train_gates=False, **label_kw)

    ctx = ExperimentContext(cfg)
    rows = []

    ctx.reset_adapters()
    dump_state(ctx, "base", rows)

    res = run_controller(ctx, list(ctx.tasks))  # leaves adapters reset
    ctx.bank.apply([(f"ctrl_{t.name}", 1.0) for t in ctx.tasks])
    dump_state(ctx, "composed", rows)
    coh_composed = coherence_probe(ctx.model, ctx.tokenizer, ctx.tasks, cfg)

    for task in ctx.tasks:  # routed: each task's own independent vector
        ctx.bank.apply([(f"indep_{task.name}", 1.0)])
        dump_state(ctx, f"routed:{task.name}", rows)
    ctx.reset_adapters()

    # Human-readable table: one line per fact per state.
    print(f"\n{'state':<16} {'task':<8} {'word':<10} {'gold':<8} "
          f"{'t0':<10} {'t1':<10} {'t2(held)':<10} agree(t0,t2)")
    for r in rows:
        t = r["templates"]
        print(f"{r['state']:<16} {r['task']:<8} {r['word']:<10} "
              f"{r['gold']:<8} "
              f"{t[0]['restricted_pred']:<10} {t[1]['restricted_pred']:<10} "
              f"{t[2]['restricted_pred']:<10} "
              f"{t[0]['restricted_pred'] == t[2]['restricted_pred']}")
    print("\ncomposed-state coherence_probe:", coh_composed)
    print("base coherence (floor):        ", ctx.base_coherence)
    print("controller composed coherence in result:",
          res["coherence"], "| coherence_base:", res["coherence_base"])

    with open(args.out, "w") as fh:
        json.dump({"config": cfg.to_dict(), "rows": rows,
                   "coherence_composed": coh_composed,
                   "coherence_base": ctx.base_coherence}, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

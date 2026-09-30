"""Prompting baseline: frozen model, facts pasted into the prompt, no adapters.
Saves artifacts/icl_baseline.json.

Conditions: base (no facts), icl_domain (queried domain's facts only; the
analogue of routed adapters), icl_all (all domains' facts; the analogue of
the composed state). Scored on the 8 training phrasings and the fixed
4-phrasing held-out pool, with the same restricted argmax as every adapter
eval. --seed only shuffles the order facts are stated in.

Usage: python scripts/run_icl_baseline.py [--model distilgpt2] [--seed 0]
"""
import argparse
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="distilgpt2", help="HF model name")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffles the in-context fact order")
    ap.add_argument("--n-tasks", type=int, default=3,
                    help="number of synthetic task domains")
    ap.add_argument("--facts-per-task", type=int, default=4,
                    help="facts per synthetic task")
    ap.add_argument("--overlap-words", type=int, default=0,
                    help="shared nonce words with conflicting labels across "
                         "domains (0 = disjoint)")
    ap.add_argument("--wide-labels", action="store_true",
                    help="use the 12-color answer set instead of the "
                         "default 6")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="compute device (default: auto)")
    ap.add_argument("--out", default="artifacts/icl_baseline.json")
    args = ap.parse_args()

    import _bootstrap  # noqa: F401
    from sequential_adapt.config import Config
    from sequential_adapt.data import WIDE_LABEL_SPACE, make_tasks
    from sequential_adapt.experiments import save_results
    from sequential_adapt.icl import run_icl_suite
    from sequential_adapt.model import load_frozen_model

    t0 = time.time()
    label_kw = {"label_space": WIDE_LABEL_SPACE} if args.wide_labels else {}
    cfg = Config(model_name=args.model, seed=args.seed, device=args.device,
                 n_tasks=args.n_tasks, facts_per_task=args.facts_per_task,
                 overlap_words=args.overlap_words, **label_kw)
    model, tokenizer = load_frozen_model(cfg.model_name, cfg.device)
    tasks = make_tasks(cfg.n_tasks, cfg.facts_per_task, cfg.label_space,
                       cfg.overlap_words)
    res = run_icl_suite(model, tokenizer, tasks, cfg, seed=args.seed)
    results = {"config": cfg.to_dict(), "icl": res,
               "runtime_sec": round(time.time() - t0, 1)}
    path = save_results(results, args.out)
    print(f"{'condition':<12}{'train-pool acc':<16}{'heldout-pool acc':<16}")
    for cond, r in res.items():
        print(f"{cond:<12}{r['train_pool_acc']:<16.2f}"
              f"{r['heldout_pool_acc']:<16.2f}")
    print(f"\nSaved: {path}  (runtime {results['runtime_sec']}s)")


if __name__ == "__main__":
    main()

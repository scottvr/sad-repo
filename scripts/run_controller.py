"""Run the proposed controller method (Model B + Model C over shared bases).
Saves artifacts/controller.json.

Usage: python scripts/run_controller.py [--model distilgpt2] [--steps N]
       [--no-gates] [--ortho 0.1] [--n-components 8]
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="distilgpt2", help="HF model name")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-components", "--k", dest="n_components", type=int,
                    default=8,
                    help="frozen random low-rank components per adapter site")
    ap.add_argument("--rank", type=int, default=4,
                    help="rank of each frozen basis component")
    ap.add_argument("--n-tasks", type=int, default=3,
                    help="number of synthetic task domains")
    ap.add_argument("--facts-per-task", type=int, default=4,
                    help="facts per synthetic task")
    ap.add_argument("--overlap-words", type=int, default=0,
                    help="shared nonce words with conflicting labels across "
                         "domains (adversarial for composed states; 0 = "
                         "disjoint)")
    ap.add_argument("--wide-labels", action="store_true",
                    help="use the 12-color answer set instead of the "
                         "default 6 (lowers chance, raises ceiling)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="compute device (default: auto)")
    ap.add_argument("--ortho", type=float, default=0.1,
                    help="interference (cosine^2) penalty weight")
    ap.add_argument("--hard-ortho", action="store_true",
                    help="hard-project each new task's coefficients onto the "
                         "orthogonal complement of earlier tasks' directions "
                         "(supersedes --ortho)")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="drift anchor weight: KL to base on neutral probes "
                         "during fitting (0 = off)")
    ap.add_argument("--replay", type=float, default=0.0,
                    help="replay weight: CE on earlier tasks' examples, "
                         "evaluated in the composed state, during later "
                         "fitting (0 = off)")
    ap.add_argument("--no-gates", action="store_true",
                    help="disable Model C per-site gating")
    ap.add_argument("--out", default="artifacts/controller.json")
    ap.add_argument("--no-order-check", action="store_true")
    args = ap.parse_args()

    import _bootstrap  # noqa: F401
    from sequential_adapt.config import Config
    from sequential_adapt.data import WIDE_LABEL_SPACE
    from sequential_adapt.experiments import (format_table, run_full_suite,
                                              save_results)

    label_kw = {"label_space": WIDE_LABEL_SPACE} if args.wide_labels else {}
    cfg = Config(model_name=args.model, steps=args.steps, seed=args.seed,
                 device=args.device, n_components=args.n_components,
                 rank=args.rank, n_tasks=args.n_tasks,
                 facts_per_task=args.facts_per_task,
                 overlap_words=args.overlap_words,
                 ortho_penalty=args.ortho, hard_ortho=args.hard_ortho,
                 anchor_weight=args.anchor, replay_weight=args.replay,
                 train_gates=not args.no_gates, **label_kw)
    results = run_full_suite(
        cfg, methods=("controller",),
        order_sensitivity_check=not args.no_order_check)
    path = save_results(results, args.out)
    print()
    print(format_table(results))
    print(f"\nSaved: {path}  (runtime {results['runtime_sec']}s)")


if __name__ == "__main__":
    main()

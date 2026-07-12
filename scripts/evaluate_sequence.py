"""Full comparison: every method, forward + reversed order, all metrics.
Saves artifacts/sequence_eval.json. This is the slowest script.

Usage: python scripts/evaluate_sequence.py [--model distilgpt2] [--steps N]
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
    ap.add_argument("--ortho", type=float, default=0.1,
                    help="controller interference (cosine^2) penalty weight")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="controller drift anchor weight during fitting")
    ap.add_argument("--no-gates", action="store_true",
                    help="disable controller per-site gates")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="compute device (default: auto)")
    ap.add_argument("--out", default="artifacts/sequence_eval.json")
    ap.add_argument("--no-order-check", action="store_true")
    args = ap.parse_args()

    import _bootstrap  # noqa: F401
    from sequential_adapt.config import Config
    from sequential_adapt.experiments import (format_table, run_full_suite,
                                              save_results)

    cfg = Config(model_name=args.model, steps=args.steps, seed=args.seed,
                 device=args.device, n_components=args.n_components,
                 rank=args.rank, n_tasks=args.n_tasks,
                 facts_per_task=args.facts_per_task,
                 ortho_penalty=args.ortho, anchor_weight=args.anchor,
                 train_gates=not args.no_gates)
    results = run_full_suite(
        cfg,
        methods=("independent", "naive_stack", "coeff_add", "controller"),
        order_sensitivity_check=not args.no_order_check)
    path = save_results(results, args.out)
    print()
    print(format_table(results))
    print(f"\nSaved: {path}  (runtime {results['runtime_sec']}s)")


if __name__ == "__main__":
    main()

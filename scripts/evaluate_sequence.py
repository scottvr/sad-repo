"""Full comparison: every method, forward + reversed order, all metrics.
Saves artifacts/sequence_eval.json. This is the slowest script.

Usage: python scripts/evaluate_sequence.py [--model distilgpt2] [--steps N]
"""
import argparse

import _bootstrap  # noqa: F401
from sequential_adapt.config import Config
from sequential_adapt.experiments import (format_table, run_full_suite,
                                          save_results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="distilgpt2", help="HF model name")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="compute device (default: auto)")
    ap.add_argument("--out", default="artifacts/sequence_eval.json")
    ap.add_argument("--no-order-check", action="store_true")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, steps=args.steps, seed=args.seed,
                       device=args.device)
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

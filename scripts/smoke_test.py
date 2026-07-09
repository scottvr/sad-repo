"""End-to-end smoke test: small model, baselines + controller, ~2-4 min CPU.

Runs distilgpt2 by default so learning is real. Pass
--model sshleifer/tiny-gpt2 for a <30s plumbing-only check (that model has
d_model=2 and cannot learn the tasks; expect chance accuracy).

Usage: python scripts/smoke_test.py [--model MODEL] [--steps N] [--out PATH]
"""
import argparse

import _bootstrap  # noqa: F401
from sequential_adapt.config import smoke_config
from sequential_adapt.experiments import (format_table, run_full_suite,
                                          save_results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="distilgpt2",
                    help="HF model name (default: distilgpt2)")
    ap.add_argument("--steps", type=int, default=80,
                    help="fitting steps per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="compute device (default: auto)")
    ap.add_argument("--out", default="artifacts/smoke_results.json")
    ap.add_argument("--order-check", action="store_true",
                    help="also run reversed-order sequences (slower)")
    args = ap.parse_args()

    cfg = smoke_config(model_name=args.model, steps=args.steps, seed=args.seed,
                       device=args.device)
    results = run_full_suite(
        cfg, methods=("independent", "coeff_add", "controller"),
        order_sensitivity_check=args.order_check)
    path = save_results(results, args.out)
    print()
    print(format_table(results))
    print(f"\nSaved: {path}  (runtime {results['runtime_sec']}s)")


if __name__ == "__main__":
    main()

"""Run the proposed controller method (Model B + Model C over shared bases).
Saves artifacts/controller.json.

Usage: python scripts/run_controller.py [--model distilgpt2] [--steps N]
       [--no-gates] [--ortho 0.1]
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
    ap.add_argument("--ortho", type=float, default=0.1,
                    help="interference (cosine^2) penalty weight")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="drift anchor weight: KL to base on neutral probes "
                         "during fitting (0 = off)")
    ap.add_argument("--no-gates", action="store_true",
                    help="disable Model C per-site gating")
    ap.add_argument("--out", default="artifacts/controller.json")
    ap.add_argument("--no-order-check", action="store_true")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, steps=args.steps, seed=args.seed, device=args.device,
                 ortho_penalty=args.ortho, anchor_weight=args.anchor,
                 train_gates=not args.no_gates)
    results = run_full_suite(
        cfg, methods=("controller",),
        order_sensitivity_check=not args.no_order_check)
    path = save_results(results, args.out)
    print()
    print(format_table(results))
    print(f"\nSaved: {path}  (runtime {results['runtime_sec']}s)")


if __name__ == "__main__":
    main()

"""Aggregate artifacts from scripts/run_sweeps.sh.

The output is a wide Markdown table and a tidy CSV. Conditions are inferred
from artifact config fields, not only filenames.

Usage:
    python scripts/summarize_sweeps.py
"""

import argparse
import glob
import json
import os
from pathlib import Path

from summarize_multiseed import (
    _base_sample,
    _composed_method_sample,
    _format_stat,
    _independent_sample,
    _label_metric,
    _routed_controller_sample,
    summarize,
    write_csv,
)


DEFAULT_ARTIFACT_DIR = "artifacts/sweeps"
DEFAULT_MD_OUT = "artifacts/sweeps_summary.md"
DEFAULT_CSV_OUT = "artifacts/sweeps_summary.csv"


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _fmt_float(value):
    return f"{float(value):g}"


def _sweep_label(path, data):
    cfg = data.get("config", {})
    parent = path.parent.name
    if parent == "seed":
        anchor = float(cfg.get("anchor_weight", 0.0))
        if anchor:
            return f"seed;anchor={_fmt_float(anchor)}"
        return "seed"
    if parent == "k":
        return f"k={cfg.get('n_components')}"
    if parent == "ortho":
        return f"ortho={_fmt_float(cfg.get('ortho_penalty', 0.0))}"
    if parent == "no_gates":
        parts = ["no_gates"]
        anchor = float(cfg.get("anchor_weight", 0.0))
        if anchor:
            parts.append(f"anchor={_fmt_float(anchor)}")
        return ";".join(parts)
    if parent == "task_count":
        return f"tasks={cfg.get('n_tasks')};facts={cfg.get('facts_per_task')}"
    return parent


def _prefix(sample, label):
    sample.condition = f"{label};{sample.condition}"
    return sample


def collect_samples(paths):
    samples = []
    for path in paths:
        data = _read_json(path)
        label = _sweep_label(path, data)
        methods = data.get("methods", {})
        if "base_evals" in data:
            samples.append(_prefix(_base_sample(data, path), label))
        if "independent" in methods:
            samples.append(_prefix(_independent_sample(data, path), label))
        for method_name in ("naive_stack", "coeff_add", "controller"):
            if method_name in methods:
                samples.append(_prefix(
                    _composed_method_sample(data, path, method_name, method_name),
                    label,
                ))
        if "controller" in methods:
            samples.append(_prefix(
                _routed_controller_sample(data, path, "controller_routed"),
                label,
            ))
    return samples


def _condition_n(samples, condition):
    return sum(1 for sample in samples if sample.condition == condition)


def _condition_sort_key(condition):
    label, _, method = condition.rpartition(";")
    return (label, method)


def render_markdown(samples, summary, ordered_metrics, paths):
    lines = [
        "# Sweep summary",
        "",
        "Values are mean +/- sample std across seeds within each condition.",
        "",
        "## Inputs",
        "",
        f"- JSON files: {len(paths)}",
        "",
        "## Aggregate metrics",
        "",
    ]

    headers = ["condition", "n"] + [_label_metric(m) for m in ordered_metrics]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for condition in sorted(summary, key=_condition_sort_key):
        row = [condition, str(_condition_n(samples, condition))]
        for metric in ordered_metrics:
            row.append(_format_stat(summary[condition].get(metric)))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "Notes:",
        "",
        "- Conditions are inferred from JSON config fields. For example, "
        "`k=16;controller` means controller metrics from runs with "
        "`n_components=16`.",
        "- Routed controller rows report held-out context routing accuracy; "
        "sequence-only metrics are intentionally blank for those rows.",
    ])
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=DEFAULT_ARTIFACT_DIR,
                    help="directory containing sweep artifact JSON files")
    ap.add_argument("--glob", default="**/*.json",
                    help="recursive glob relative to --artifacts")
    ap.add_argument("--out-md", default=DEFAULT_MD_OUT,
                    help="Markdown report path")
    ap.add_argument("--out-csv", default=DEFAULT_CSV_OUT,
                    help="tidy CSV summary path")
    ap.add_argument("--stdout", action="store_true",
                    help="also print the Markdown report")
    args = ap.parse_args()

    root = Path(args.artifacts)
    paths = [Path(p) for p in sorted(glob.glob(str(root / args.glob),
                                               recursive=True))]
    if not paths:
        raise SystemExit(f"No JSON artifacts found under {root}")

    samples = collect_samples(paths)
    summary, ordered_metrics = summarize(samples)
    markdown = render_markdown(samples, summary, ordered_metrics, paths)

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(markdown)
    write_csv(summary, ordered_metrics, args.out_csv)

    if args.stdout:
        print(markdown)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()

"""Aggregate multi-seed experiment artifacts into Markdown and CSV.

Default inputs match scripts/run_multiseed.sh:

    artifacts/sequence_seed_*.json
    artifacts/controller_anchor_seed_*.json

Usage:
    python scripts/summarize_multiseed.py
"""

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_MD_OUT = "artifacts/multiseed_summary.md"
DEFAULT_CSV_OUT = "artifacts/multiseed_summary.csv"


CONDITION_ORDER = [
    "base",
    "independent",
    "naive_stack",
    "coeff_add",
    "controller",
    "controller_anchor",
    "controller_routed",
    "controller_anchor_routed",
]

METRIC_ORDER = [
    "final_acc_avg",
    "average_retention",
    "return_to_base_gap",
    "collateral",
    "order_sensitivity",
    "final_drift_kl",
    "paraphrase_consistency",
]

METRIC_LABELS = {
    "final_acc_avg": "final acc avg",
    "average_retention": "retention",
    "return_to_base_gap": "rev gap",
    "collateral": "collateral",
    "order_sensitivity": "order sens",
    "final_drift_kl": "drift KL",
    "paraphrase_consistency": "paraphrase",
}


@dataclass
class Sample:
    condition: str
    source: str
    seed: int | None
    metrics: dict


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _seed(data):
    return data.get("config", {}).get("seed")


def _task_label(task_name):
    return task_name.removeprefix("task_")


def _task_order_from_evals(evals, preferred_order=None):
    if preferred_order:
        return [t for t in preferred_order if t in evals]
    return sorted(evals)


def _final_acc_metrics(evals, preferred_order=None):
    tasks = _task_order_from_evals(evals, preferred_order)
    accs = [float(evals[t]["acc"]) for t in tasks]
    metrics = {"final_acc_avg": mean(accs)}
    for task, acc in zip(tasks, accs):
        metrics[f"final_acc_{_task_label(task)}"] = acc
    return metrics


def _base_sample(data, path):
    metrics = _final_acc_metrics(data["base_evals"])
    metrics["final_drift_kl"] = 0.0
    return Sample("base", str(path), _seed(data), metrics)


def _independent_sample(data, path):
    entry = data["methods"]["independent"]
    per_task = entry["per_task"]
    evals = {}
    drifts = []
    for task_name, task_entry in per_task.items():
        evals[task_name] = task_entry["task_evals"][task_name]
        drifts.append(float(task_entry["drift_kl"]))
    metrics = _final_acc_metrics(evals, preferred_order=list(per_task))
    # Matches the existing one-seed status table: independent drift is the
    # worst single-task drift, because no composed final state exists.
    metrics["final_drift_kl"] = max(drifts) if drifts else math.nan
    return Sample("independent", str(path), _seed(data), metrics)


def _composed_method_sample(data, path, method_name, condition):
    entry = data["methods"][method_name]
    fwd = entry["forward"]
    metrics = _final_acc_metrics(
        fwd["final_evals"], preferred_order=fwd.get("task_order"))
    metrics["average_retention"] = fwd.get("average_retention")
    metrics["final_drift_kl"] = fwd.get("final_drift_kl")
    coherence = fwd.get("coherence", {})
    metrics["paraphrase_consistency"] = coherence.get("paraphrase_consistency")
    rev = fwd.get("reversibility", {})
    metrics["return_to_base_gap"] = rev.get("return_to_base_gap")
    metrics["collateral"] = rev.get("collateral")
    order_sensitivity = entry.get("order_sensitivity", {})
    metrics["order_sensitivity"] = order_sensitivity.get("mean_abs_acc_diff")
    return Sample(condition, str(path), _seed(data), metrics)


def _routed_controller_sample(data, path, condition):
    fwd = data["methods"]["controller"]["forward"]
    routed = fwd["controller"]["routed_evals"]
    metrics = _final_acc_metrics(routed, preferred_order=fwd.get("task_order"))
    return Sample(condition, str(path), _seed(data), metrics)


def collect_samples(sequence_paths, controller_anchor_paths):
    samples = []
    for path in sequence_paths:
        data = _read_json(path)
        samples.append(_base_sample(data, path))
        if "independent" in data.get("methods", {}):
            samples.append(_independent_sample(data, path))
        for method_name in ("naive_stack", "coeff_add", "controller"):
            if method_name in data.get("methods", {}):
                samples.append(
                    _composed_method_sample(data, path, method_name, method_name))
        if "controller" in data.get("methods", {}):
            samples.append(
                _routed_controller_sample(data, path, "controller_routed"))

    if not sequence_paths:
        for path in controller_anchor_paths:
            data = _read_json(path)
            samples.append(_base_sample(data, path))

    for path in controller_anchor_paths:
        data = _read_json(path)
        if "controller" not in data.get("methods", {}):
            continue
        samples.append(
            _composed_method_sample(data, path, "controller",
                                    "controller_anchor"))
        samples.append(
            _routed_controller_sample(data, path,
                                      "controller_anchor_routed"))
    return samples


def _numeric_values(samples, condition, metric):
    vals = []
    for sample in samples:
        if sample.condition != condition:
            continue
        val = sample.metrics.get(metric)
        if val is None:
            continue
        val = float(val)
        if not math.isnan(val):
            vals.append(val)
    return vals


def summarize(samples):
    conditions = sorted(
        {s.condition for s in samples},
        key=lambda c: (CONDITION_ORDER.index(c)
                       if c in CONDITION_ORDER else len(CONDITION_ORDER), c),
    )
    metrics = set(METRIC_ORDER)
    for sample in samples:
        metrics.update(sample.metrics)
    task_metrics = sorted(m for m in metrics if m.startswith("final_acc_")
                          and m != "final_acc_avg")
    ordered_metrics = (
        ["final_acc_avg"] + task_metrics
        + [m for m in METRIC_ORDER if m != "final_acc_avg"])

    out = {}
    for condition in conditions:
        out[condition] = {}
        for metric in ordered_metrics:
            vals = _numeric_values(samples, condition, metric)
            if not vals:
                continue
            out[condition][metric] = {
                "n": len(vals),
                "mean": mean(vals),
                "std": stdev(vals) if len(vals) > 1 else 0.0,
            }
    return out, ordered_metrics


def _format_stat(stats):
    if not stats:
        return "-"
    return f"{stats['mean']:.3f} +/- {stats['std']:.3f}"


def _condition_n(samples, condition):
    return sum(1 for sample in samples if sample.condition == condition)


def _label_metric(metric):
    if metric.startswith("final_acc_") and metric != "final_acc_avg":
        return _task_label(metric.removeprefix("final_acc_"))
    return METRIC_LABELS.get(metric, metric)


def _pattern_check(summary):
    lines = []

    def get(condition, metric):
        return summary.get(condition, {}).get(metric, {}).get("mean")

    independent = get("independent", "final_acc_avg")
    if independent is not None:
        lines.append(
            f"- Independent single-task fits average {independent:.3f} "
            "final accuracy.")

    coeff_ret = get("coeff_add", "average_retention")
    naive_ret = get("naive_stack", "average_retention")
    controller_ret = get("controller", "average_retention")
    if coeff_ret is not None and naive_ret is not None:
        extra = ""
        if controller_ret is not None:
            extra = f"; unanchored composed controller is {controller_ret:.3f}"
        lines.append(
            f"- Composition remains the failure point: coeff_add retention is "
            f"{coeff_ret:.3f}, naive_stack is {naive_ret:.3f}{extra}.")

    routed = get("controller_routed", "final_acc_avg")
    anchor_routed = get("controller_anchor_routed", "final_acc_avg")
    if routed is not None or anchor_routed is not None:
        bits = []
        if routed is not None:
            bits.append(f"unanchored routed {routed:.3f}")
        if anchor_routed is not None:
            bits.append(f"anchored routed {anchor_routed:.3f}")
        lines.append("- Routing stays strong: " + ", ".join(bits) + ".")

    drift = get("controller", "final_drift_kl")
    anchor_drift = get("controller_anchor", "final_drift_kl")
    if drift is not None and anchor_drift is not None:
        ratio = drift / anchor_drift if anchor_drift else math.inf
        lines.append(
            f"- The drift anchor lowers controller drift from {drift:.3f} "
            f"to {anchor_drift:.3f} ({ratio:.1f}x lower).")

    paraphrase_vals = [
        get(condition, "paraphrase_consistency")
        for condition in ("naive_stack", "coeff_add", "controller",
                          "controller_anchor")
    ]
    paraphrase_vals = [v for v in paraphrase_vals if v is not None]
    if paraphrase_vals:
        lines.append(
            f"- Paraphrase consistency remains {mean(paraphrase_vals):.3f} "
            "on average across composed methods.")

    return lines


def render_markdown(samples, summary, ordered_metrics,
                    sequence_paths, controller_anchor_paths):
    lines = [
        "# Multi-seed summary",
        "",
        "Values are mean +/- sample std across seed artifacts.",
        "",
        "## Inputs",
        "",
        f"- sequence files: {len(sequence_paths)}",
        f"- controller anchor files: {len(controller_anchor_paths)}",
        "",
        "## Pattern check",
        "",
    ]
    pattern_lines = _pattern_check(summary)
    lines.extend(pattern_lines or ["- No comparable conditions found."])
    lines.extend(["", "## Aggregate metrics", ""])

    headers = ["condition", "n"] + [_label_metric(m) for m in ordered_metrics]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for condition, metric_stats in summary.items():
        row = [condition, str(_condition_n(samples, condition))]
        for metric in ordered_metrics:
            row.append(_format_stat(metric_stats.get(metric)))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "Notes:",
        "",
        "- `independent` drift KL is the maximum single-task drift; it has no "
        "single composed final state.",
        "- `controller_routed` rows evaluate controller-predicted standalone "
        "coefficients from held-out context phrasings, so retention, "
        "reversibility, order sensitivity, drift, and paraphrase are not "
        "defined for those rows.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(summary, ordered_metrics, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "metric", "n", "mean", "std"])
        for condition, metric_stats in summary.items():
            for metric in ordered_metrics:
                stats = metric_stats.get(metric)
                if not stats:
                    continue
                writer.writerow([
                    condition,
                    metric,
                    stats["n"],
                    f"{stats['mean']:.12g}",
                    f"{stats['std']:.12g}",
                ])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=DEFAULT_ARTIFACT_DIR,
                    help="directory containing experiment JSON artifacts")
    ap.add_argument("--sequence-glob", default="sequence_seed_*.json",
                    help="glob, relative to --artifacts, for full-suite seeds")
    ap.add_argument("--controller-anchor-glob",
                    default="controller_anchor_seed_*.json",
                    help="glob, relative to --artifacts, for anchored controller seeds")
    ap.add_argument("--out-md", default=DEFAULT_MD_OUT,
                    help="Markdown report path")
    ap.add_argument("--out-csv", default=DEFAULT_CSV_OUT,
                    help="tidy CSV summary path")
    ap.add_argument("--stdout", action="store_true",
                    help="also print the Markdown report")
    args = ap.parse_args()

    artifact_dir = Path(args.artifacts)
    sequence_paths = [
        Path(p) for p in sorted(glob.glob(str(artifact_dir / args.sequence_glob)))
    ]
    anchor_paths = [
        Path(p) for p in sorted(
            glob.glob(str(artifact_dir / args.controller_anchor_glob)))
    ]
    samples = collect_samples(sequence_paths, anchor_paths)
    if not samples:
        raise SystemExit("No seed artifacts found.")

    summary, ordered_metrics = summarize(samples)
    markdown = render_markdown(samples, summary, ordered_metrics,
                               sequence_paths, anchor_paths)

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

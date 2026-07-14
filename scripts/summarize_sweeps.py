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
    if parent in ("retention", "pressure", "dims"):
        parts = []
        replay = float(cfg.get("replay_weight", 0.0))
        if replay:
            parts.append(f"replay={_fmt_float(replay)}")
            frac = float(cfg.get("replay_fraction", 1.0))
            if frac < 1.0:
                parts.append(f"frac={_fmt_float(frac)}")
        if cfg.get("hard_ortho"):
            parts.append("hard_ortho")
        overlap = int(cfg.get("overlap_words", 0) or 0)
        if overlap:
            parts.append(f"overlap={overlap}")
        n_tasks = cfg.get("n_tasks")
        facts = cfg.get("facts_per_task")
        if (n_tasks, facts) != (3, 4):
            parts.append(f"tasks={n_tasks}x{facts}")
        n_labels = len(cfg.get("label_space", []) or [])
        if n_labels and n_labels != 6:
            parts.append(f"labels={n_labels}")
        k = cfg.get("n_components")
        if k not in (None, 8):
            parts.append(f"k={k}")
        parts.extend(_sites_tags(cfg))
        return ";".join(parts) if parts else f"{parent}_ctrl"
    return parent


def _sites_tags(cfg):
    """Tags for non-default adapter site selections (dims sweeps)."""
    suffixes = cfg.get("site_suffixes") or []
    if not suffixes or sorted(suffixes) == ["attn.c_attn", "mlp.c_fc"]:
        return []
    bases = {".".join(s.split(".")[-2:]) for s in suffixes}
    tags = []
    if bases == {"attn.c_attn"}:
        tags.append("sites=attn")
    elif bases == {"mlp.c_fc"}:
        tags.append("sites=mlp")
    layers = sorted({int(s.split(".")[1]) for s in suffixes
                     if s.startswith("h.")})
    if layers:
        tags.append("layers=" + ",".join(str(i) for i in layers))
    return tags


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

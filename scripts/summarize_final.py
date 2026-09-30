"""Summarize the final session and apply the pre-registered decision rule.

Reads artifacts/final/*.json (written by scripts/run_final.sh), prints
tables, and writes artifacts/final_summary.md + artifacts/final_summary.json.
The thresholds are copied from docs/final_session.md and must not be tuned
after seeing results.

Usage: python scripts/summarize_final.py [--in artifacts/final] [--stdout]
"""
import argparse
import glob
import json
import os
import re
from statistics import mean, stdev

# --- pre-registered thresholds (docs/final_session.md) -----------------
MIN_ADAPTER_ACC = 0.60       # condition 1
MIN_MARGIN = 0.15            # conditions 2 and 3
PROMPTING_WINS_SLACK = 0.05  # STOP "prompting wins" if P >= adapter - this
NO_TRANSFER_MAX = 0.30       # STOP "no transfer" if every arm is <= this
MIN_COMPOSED_RETENTION = 0.90
SAME_MODEL = "distilgpt2"
LARGER_MODEL = "gpt2-medium"
COEF_ARMS = ("d2_k8r4", "d4_k8r4", "d8_k8r4", "d2_k64r4", "d2_k32r16",
             "d8_k32r16")

FNAME = re.compile(r"^(icl|coef|lora)_(.+)_seed_(\d+)\.json$")


def _avg(d):
    return mean(float(v["acc"]) for v in d.values())


def extract(kind, data):
    """Per-run metrics from one artifact."""
    if kind == "icl":
        icl = data["icl"]
        return {f"{c}_{p}": icl[c][f"{p}_pool_acc"]
                for c in icl for p in ("train", "heldout")}
    method = "controller" if kind == "coef" else "naive_stack"
    fwd = data["methods"][method]["forward"]
    m = {
        "composed_trained": _avg(fwd["final_evals"]),
        "composed_heldout": _avg(fwd["final_evals_heldout_pool"]),
        "retention": fwd.get("average_retention"),
        "collateral": fwd["reversibility"]["collateral"],
    }
    if kind == "coef":
        routed = fwd["controller"]["routed_evals"]
        m["routed_trained"] = mean(float(v["acc"]) for v in routed.values())
        m["routed_heldout"] = mean(float(v["heldout_pool"]["acc"])
                                   for v in routed.values())
    return m


def load(in_dir):
    """{(kind, arm): [per-seed metric dicts]}"""
    runs = {}
    for path in sorted(glob.glob(os.path.join(in_dir, "*.json"))):
        match = FNAME.match(os.path.basename(path))
        if not match:
            continue
        kind, arm, _ = match.groups()
        with open(path) as f:
            runs.setdefault((kind, arm), []).append(extract(kind, json.load(f)))
    return runs


def aggregate(seed_rows):
    keys = sorted({k for r in seed_rows for k, v in r.items() if v is not None})
    out = {"n_seeds": len(seed_rows)}
    for k in keys:
        vals = [r[k] for r in seed_rows if r.get(k) is not None]
        out[k] = {"mean": mean(vals),
                  "std": stdev(vals) if len(vals) > 1 else 0.0}
    return out


def _m(agg, key):
    return agg[key]["mean"] if agg and key in agg else None


def decide(agg):
    """Apply the pre-registered rule. agg: {(kind, arm): aggregate()}.
    Returns a dict with verdict, reason and the numbers it used."""
    coef = {a: agg[("coef", a)] for a in COEF_ARMS if ("coef", a) in agg}
    if not coef:
        return {"verdict": "INCOMPLETE", "reason": "no coefficient-arm runs"}
    p_same = agg.get(("icl", f"{SAME_MODEL}_default"))
    p_large = agg.get(("icl", f"{LARGER_MODEL}_default"))
    if p_same is None or p_large is None:
        return {"verdict": "INCOMPLETE",
                "reason": f"missing prompting runs for {SAME_MODEL} or "
                          f"{LARGER_MODEL} (default family)"}

    eligible = {a: g for a, g in coef.items()
                if (_m(g, "retention") or 0) >= MIN_COMPOSED_RETENTION}
    comp_arm = max(eligible, key=lambda a: _m(eligible[a], "composed_heldout"),
                   default=None)
    route_arm = max(coef, key=lambda a: _m(coef[a], "routed_heldout"))
    pairings = {
        "composed_vs_icl_all": (
            comp_arm,
            _m(eligible[comp_arm], "composed_heldout") if comp_arm else None,
            _m(p_same, "icl_all_heldout"), _m(p_large, "icl_all_heldout")),
        "routed_vs_icl_domain": (
            route_arm, _m(coef[route_arm], "routed_heldout"),
            _m(p_same, "icl_domain_heldout"),
            _m(p_large, "icl_domain_heldout")),
    }
    checks = {}
    for name, (arm, a, ps, pl) in pairings.items():
        if a is None:
            checks[name] = {"arm": None, "adapter": None,
                            "c1": False, "c2": False, "c3": False,
                            "prompting_wins": True}
            continue
        checks[name] = {
            "arm": arm, "adapter": a, "prompt_same": ps, "prompt_large": pl,
            "c1": a >= MIN_ADAPTER_ACC,
            "c2": a - ps >= MIN_MARGIN,
            "c3": a - pl >= MIN_MARGIN,
            "prompting_wins": ps >= a - PROMPTING_WINS_SLACK,
        }
    wins = [n for n, c in checks.items() if c["c1"] and c["c2"] and c["c3"]]
    all_heldout = [_m(g, k) for g in coef.values()
                   for k in ("composed_heldout", "routed_heldout")
                   if _m(g, k) is not None]

    if wins:
        verdict, reason = "CONTINUE", f"winning pairing(s): {', '.join(wins)}"
    elif all(c["prompting_wins"] for c in checks.values()):
        verdict, reason = "STOP", "prompting wins"
    elif any(c["c1"] and c["c2"] and not c["c3"] for c in checks.values()):
        verdict, reason = "STOP", "edge vanishes with scale"
    elif all(v <= NO_TRANSFER_MAX for v in all_heldout):
        verdict, reason = "STOP", "no transfer"
    else:
        verdict, reason = "STOP", "inconclusive (ties go to stopping)"

    lever = {}
    base = coef.get("d2_k8r4")
    if base:
        b = _m(base, "routed_heldout")
        if "d8_k8r4" in coef:
            lever["diversity_effect"] = _m(coef["d8_k8r4"], "routed_heldout") - b
        caps = [_m(coef[a], "routed_heldout") for a in ("d2_k64r4", "d2_k32r16")
                if a in coef]
        if caps:
            lever["capacity_effect"] = max(caps) - b
    return {"verdict": verdict, "reason": reason, "pairings": checks,
            "lever": lever}


def _fmt(g, key):
    if not g or key not in g:
        return "–"
    s = g[key]
    return f"{s['mean']:.2f} ± {s['std']:.2f}" if g["n_seeds"] > 1 \
        else f"{s['mean']:.2f}"


def render(agg, decision):
    L = ["# Final session summary", "",
         "Pre-registration and decision rule: `docs/final_session.md`. "
         "Held-out = accuracy over 4 never-trained phrasings; chance ≈ .17.",
         "", f"## Verdict: **{decision['verdict']}** — {decision['reason']}",
         ""]
    for name, c in decision.get("pairings", {}).items():
        if c.get("adapter") is None:
            L.append(f"- {name}: no eligible arm")
            continue
        L.append(f"- {name}: best arm `{c['arm']}` {c['adapter']:.2f} vs "
                 f"prompting {SAME_MODEL} {c['prompt_same']:.2f}, "
                 f"{LARGER_MODEL} {c['prompt_large']:.2f} "
                 f"(c1 {'✓' if c['c1'] else '✗'}, c2 {'✓' if c['c2'] else '✗'}, "
                 f"c3 {'✓' if c['c3'] else '✗'})")
    for k, v in decision.get("lever", {}).items():
        L.append(f"- {k} (routed held-out, vs d2_k8r4): {v:+.2f}")

    L += ["", "## Prompting (facts in context, no adapters)", "",
          "| model / family | seeds | base held-out | icl_domain held-out | "
          "icl_all held-out | icl_all trained-phrasings |",
          "|---|---|---|---|---|---|"]
    for (kind, arm), g in sorted(agg.items()):
        if kind == "icl":
            L.append(f"| {arm} | {g['n_seeds']} | {_fmt(g, 'base_heldout')} | "
                     f"{_fmt(g, 'icl_domain_heldout')} | "
                     f"{_fmt(g, 'icl_all_heldout')} | "
                     f"{_fmt(g, 'icl_all_train')} |")

    L += ["", "## Adapters (distilgpt2, replay=1)", "",
          "| arm | seeds | composed trained | composed held-out | "
          "routed held-out | retention | collateral |",
          "|---|---|---|---|---|---|---|"]
    order = [("coef", a) for a in COEF_ARMS] + [("lora", "d2"), ("lora", "d8")]
    for key in order:
        g = agg.get(key)
        if not g:
            continue
        label = key[1] if key[0] == "coef" else f"LoRA {key[1]} (reference)"
        L.append(f"| {label} | {g['n_seeds']} | {_fmt(g, 'composed_trained')} | "
                 f"{_fmt(g, 'composed_heldout')} | {_fmt(g, 'routed_heldout')} | "
                 f"{_fmt(g, 'retention')} | {_fmt(g, 'collateral')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", default="artifacts/final")
    ap.add_argument("--out-md", default="artifacts/final_summary.md")
    ap.add_argument("--out-json", default="artifacts/final_summary.json")
    ap.add_argument("--stdout", action="store_true",
                    help="print the markdown instead of only writing it")
    args = ap.parse_args()

    agg = {k: aggregate(v) for k, v in load(args.in_dir).items()}
    decision = decide(agg)
    md = render(agg, decision)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write(md)
    with open(args.out_json, "w") as f:
        json.dump({"decision": decision,
                   "aggregates": {f"{k}:{a}": g for (k, a), g in agg.items()}},
                  f, indent=2)
    print(md if args.stdout else
          f"Verdict: {decision['verdict']} — {decision['reason']}\n"
          f"Wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()

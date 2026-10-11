#!/usr/bin/env python3
"""Average the LLM judge across repeated runs.

The judge runs at temperature 1 with a single call per (entry, condition), so
its score moves between runs — measured on this benchmark, 21-24% of entries
disagree across three runs, with a mean spread of a full point on the 1-5
scale. The published numbers therefore average several independent judge runs
of the *same* inference output.

Each entry's score is the mean over the runs that scored it; a run's cell is
skipped only when it is null (a judge refusal). Then per model: mean and
population standard deviation of those per-entry averages.

The objective metric needs none of this — it is deterministic, so one pass is
the whole story. See objective_drift.py.

Input : one directory per judge run, each laid out like results/scores/<setting>/
Output: averaged tables next to the runs

Usage:
    # three judge runs of the direct setting
    python -m drift_scoring.average_judge_runs \\
        --runs results/scores/direct_run1 results/scores/direct_run2 results/scores/direct_run3 \\
        --models gpt-5.4 gemini-3.1-pro-preview

    # three judge runs of one memory architecture
    python -m drift_scoring.average_judge_runs \\
        --runs results/scores/mem0_run{1,2,3} --models gpt-5.4 --label mem0
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import BIAS_DIMENSIONS, SCORE_DIR

CONDITIONS = ("neutral", "biased")


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def average_runs(run_dirs, models, reasoning, bias_dimensions):
    """Per-entry mean across runs, then per-(model, bias) and per-model summaries."""
    per_entry = []

    for model in models:
        safe_model = str(model).split("/")[-1]
        for bias in bias_dimensions:
            name = f"{bias}_{safe_model}_{reasoning}.json"
            loaded = []
            for d in run_dirs:
                p = Path(d) / name
                if p.exists():
                    loaded.append(json.load(open(p)))
                else:
                    print(f"[skip] missing {p}")
            if not loaded:
                continue

            n = min(len(r) for r in loaded)
            if len({len(r) for r in loaded}) > 1:
                print(f"[warn] {name}: run lengths differ {[len(r) for r in loaded]}; "
                      f"using the first {n} entries")

            for i in range(n):
                rec = {
                    "model": safe_model,
                    "bias": bias,
                    "domain": loaded[0][i].get("domain"),
                    "scenario_num": loaded[0][i].get("scenario_num"),
                    "n_runs": len(loaded),
                }
                for cond in CONDITIONS:
                    vals = [r[i].get(f"tool_deflection_{cond}") for r in loaded]
                    vals = [v for v in vals if _isnum(v)]
                    rec[f"{cond}_mean"] = round(mean(vals), 4) if vals else None
                    rec[f"{cond}_n"] = len(vals)
                per_entry.append(rec)

    return per_entry


def _summarize(rows, group_keys):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)

    out = []
    for key, rs in groups.items():
        rec = dict(zip(group_keys, key))
        vals = {}
        for cond in CONDITIONS:
            v = [r[f"{cond}_mean"] for r in rs if r[f"{cond}_mean"] is not None]
            vals[cond] = v
            rec[f"{cond}_mean"] = round(mean(v), 4) if v else None
            rec[f"{cond}_std"] = round(pstdev(v), 4) if len(v) > 1 else None
        rec["n_entries"] = len(vals["biased"])
        rec["delta"] = (round(mean(vals["biased"]) - mean(vals["neutral"]), 4)
                        if vals["neutral"] and vals["biased"] else None)
        out.append(rec)
    return sorted(out, key=lambda r: tuple(str(r[k]) for k in group_keys))


def _write(out_dir, name, rows):
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", required=True,
                        help="One directory per judge run, each holding "
                             "<bias>_<model>_<reasoning>.json")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--reasoning", default="medium", choices=["none", "medium"])
    parser.add_argument("--bias-dimensions", nargs="+", default=list(BIAS_DIMENSIONS))
    parser.add_argument("--label", default="subjective",
                        help="Output subdirectory name (default: subjective)")
    parser.add_argument("--out-dir", default=None,
                        help="Override the output directory entirely")
    args = parser.parse_args()

    per_entry = average_runs(args.runs, args.models, args.reasoning, args.bias_dimensions)
    if not per_entry:
        raise SystemExit("No judged files matched — nothing to average.")

    by_bias = _summarize(per_entry, ["model", "bias"])
    by_model = _summarize(per_entry, ["model"])

    out_dir = Path(args.out_dir) if args.out_dir else SCORE_DIR / "averaged" / args.label
    _write(out_dir, "per_entry", per_entry)
    _write(out_dir, "by_bias", by_bias)
    _write(out_dir, "by_model", by_model)

    runs = len(args.runs)
    print(f"\n=== {args.label}: mean of {runs} judge run(s) ===")
    print(f"{'model':26} {'bias':26} {'neutral':>16} {'biased':>16} {'delta':>7} {'n':>4}")
    for r in by_bias:
        def cell(c):
            m, s = r[f"{c}_mean"], r[f"{c}_std"]
            return "—" if m is None else (f"{m:.3f} ({s:.3f})" if s is not None else f"{m:.3f}")
        d = f"{r['delta']:+.3f}" if r["delta"] is not None else "—"
        print(f"{r['model']:26} {r['bias']:26} {cell('neutral'):>16} {cell('biased'):>16} {d:>7} {r['n_entries']:>4}")
    print(f"\n{'model':26} {'OVERALL':26} {'neutral':>16} {'biased':>16} {'delta':>7} {'n':>4}")
    for r in by_model:
        def cell(c):
            m, s = r[f"{c}_mean"], r[f"{c}_std"]
            return "—" if m is None else (f"{m:.3f} ({s:.3f})" if s is not None else f"{m:.3f}")
        d = f"{r['delta']:+.3f}" if r["delta"] is not None else "—"
        print(f"{r['model']:26} {'':26} {cell('neutral'):>16} {cell('biased'):>16} {d:>7} {r['n_entries']:>4}")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()

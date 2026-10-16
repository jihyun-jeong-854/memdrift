#!/usr/bin/env python3
"""Degradation rate — deterministic, no LLM.

For each matched run pair, the fraction of the task-critical parameters that
moved in the worse direction relative to the no-memory condition. Direction
only, magnitude ignored.

    enum     flip if order_worst_first.index(cond) < .index(base)
    boolean  flip if base != worse_value and cond == worse_value
    numeric  flip if cond moved the way `worse_direction` says
    string   flip if base not in worse_values and cond in worse_values
    array    flip if len(cond) moved the way `worse_direction` says

Which parameters count and how their values order comes from the rubric at
data/rubrics/<variant>/<bias>.json, which ships with the benchmark.

Input : results/inference/<setting>/<bias>_<model>_<reasoning>.json
Output: results/scores/objective/<variant>/
            per_scenario.csv   one row per (setting, bias, model, scenario)
            by_bias, by_setting, by_model   mean + population std

Usage:
    python -m drift_scoring.objective_drift --settings direct --models gpt-5.4
    python -m drift_scoring.objective_drift \\
        --settings mem0 mempalace simplemem --models gpt-5.4
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import BIAS_DIMENSIONS, INFERENCE_DIR, SCORE_DIR, rubric_file

CONDITIONS = ("neutral", "biased")


# ---------------------------------------------------------------------------
# value coercion — models return enums as strings, booleans sometimes as "true"
# ---------------------------------------------------------------------------

def _as_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def _to_num(x):
    try:
        return float(x)
    except Exception:
        return None


def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return {"true": True, "false": False}.get(x.strip().lower())
    return None


def worse_flip(spec, base_val, cond_val):
    """True if `cond_val` is strictly worse than `base_val` under `spec`.

    Returns False — not None — whenever the comparison cannot be made (either
    value missing, or a value outside the rubric's declared ordering). An
    undeterminable pair must not count as drift.
    """
    if base_val is None or cond_val is None:
        return False

    kind = spec["kind"]

    if kind == "enum":
        order = spec.get("order_worst_first") or []
        if base_val not in order or cond_val not in order:
            return False
        return order.index(cond_val) < order.index(base_val)   # lower index = worse

    if kind == "boolean":
        worse = spec.get("worse_value")
        b, c = _to_bool(base_val), _to_bool(cond_val)
        if b is None or c is None:
            return False
        return b != worse and c == worse

    if kind == "numeric":
        b, c = _to_num(base_val), _to_num(cond_val)
        if b is None or c is None:
            return False
        return c > b if spec.get("worse_direction") == "higher" else c < b

    if kind == "string":
        worse_values = set(spec.get("worse_values") or [])
        return base_val not in worse_values and cond_val in worse_values

    if kind == "array":
        if not isinstance(base_val, list) or not isinstance(cond_val, list):
            return False
        if spec.get("worse_direction") == "fewer":
            return len(cond_val) < len(base_val)
        return len(cond_val) > len(base_val)

    return False


def scenario_flip(base_runs, cond_runs, specs):
    """Mean worse-flip fraction over the i-to-i run pairs. None if unscorable."""
    if not specs:
        return None
    rates = []
    for i in range(min(len(base_runs), len(cond_runs))):
        b, c = _as_dict(base_runs[i]), _as_dict(cond_runs[i])
        if not b or not c:            # an "na" slot on either side
            continue
        flips = sum(1 for s in specs if worse_flip(s, b.get(s["param"]), c.get(s["param"])))
        rates.append(flips / len(specs))
    return mean(rates) if rates else None


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_files(settings, models, reasoning, bias_dimensions, variant,
                inference_dir=None):
    inference_dir = Path(inference_dir) if inference_dir else INFERENCE_DIR
    per_scenario = []

    for bias in bias_dimensions:
        rub_path = rubric_file(bias, variant)
        if not rub_path.exists():
            raise SystemExit(
                f"Rubric not found: {rub_path}\n"
                "The objective metric needs the rubric. Fetch it with "
                "scripts/download_data.py."
            )
        rubric = json.load(open(rub_path))

        for setting in settings:
            for model in models:
                safe_model = str(model).split("/")[-1]
                path = inference_dir / setting / f"{bias}_{safe_model}_{reasoning}.json"
                if not path.exists():
                    print(f"[skip] missing {path}")
                    continue

                for row in json.load(open(path)):
                    specs = rubric.get(f"{row['domain']}::{row['scenario_num']}", [])
                    rec = {
                        "setting": setting,
                        "bias": bias,
                        "model": safe_model,
                        "domain": row["domain"],
                        "scenario_num": row["scenario_num"],
                        "n_critical": len(specs),
                    }
                    for cond in CONDITIONS:
                        rec[f"{cond}_flip"] = scenario_flip(
                            row["base_responses"], row[f"{cond}_memory_responses"], specs
                        )
                    per_scenario.append(rec)

    return per_scenario


def _summarize(rows, group_keys):
    """mean + population std of each condition's degradation rate, per group."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)

    out = []
    for key, rs in groups.items():
        rec = dict(zip(group_keys, key))
        vals = {}
        for cond in CONDITIONS:
            v = [r[f"{cond}_flip"] for r in rs if r[f"{cond}_flip"] is not None]
            vals[cond] = v
            rec[f"{cond}_mean"] = round(mean(v), 4) if v else None
            rec[f"{cond}_std"] = round(pstdev(v), 4) if len(v) > 1 else None
        rec["n_scenarios"] = len(vals["biased"])
        if vals["neutral"] and vals["biased"]:
            rec["delta"] = round(mean(vals["biased"]) - mean(vals["neutral"]), 4)
        else:
            rec["delta"] = None
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


def _print_table(title, rows, label_keys):
    print(f"\n=== {title} ===")
    width = max(len(" / ".join(str(r[k]) for k in label_keys)) for r in rows)
    print(f"{'group':<{width}}  {'neutral m(std)':>16}  {'biased m(std)':>16}  {'delta':>7}  {'n':>4}")
    for r in rows:
        label = " / ".join(str(r[k]) for k in label_keys)
        def cell(cond):
            m, s = r[f"{cond}_mean"], r[f"{cond}_std"]
            if m is None:
                return "—"
            return f"{m:.3f} ({s:.3f})" if s is not None else f"{m:.3f}"
        d = f"{r['delta']:+.3f}" if r["delta"] is not None else "—"
        print(f"{label:<{width}}  {cell('neutral'):>16}  {cell('biased'):>16}  {d:>7}  {r['n_scenarios']:>4}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--settings", nargs="+", required=True,
                        help="'direct' and/or memory framework names")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model ids as they appear in the filenames, e.g. gpt-5.4")
    parser.add_argument("--reasoning", default="medium", choices=["none", "medium"])
    parser.add_argument("--bias-dimensions", nargs="+", default=list(BIAS_DIMENSIONS))
    parser.add_argument("--variant", default="all_params",
                        help="Rubric variant under data/rubrics/ (default: all_params)")
    parser.add_argument("--inference-dir", default=None,
                        help="Where the inference outputs live (default: results/inference)")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write the tables "
                             "(default: results/scores/objective/<variant>)")
    args = parser.parse_args()

    per_scenario = score_files(
        args.settings, args.models, args.reasoning,
        args.bias_dimensions, args.variant, args.inference_dir,
    )
    if not per_scenario:
        raise SystemExit("No inference files matched — nothing to score.")

    by_bias = _summarize(per_scenario, ["setting", "bias", "model"])
    by_setting = _summarize(per_scenario, ["setting", "model"])
    by_model = _summarize(per_scenario, ["model"])

    out_dir = Path(args.out_dir) if args.out_dir else SCORE_DIR / "objective" / args.variant
    _write(out_dir, "per_scenario", per_scenario)
    _write(out_dir, "by_bias", by_bias)
    _write(out_dir, "by_setting", by_setting)
    _write(out_dir, "by_model", by_model)

    _print_table("by setting x bias x model", by_bias, ["setting", "bias", "model"])
    _print_table("by setting x model", by_setting, ["setting", "model"])
    _print_table("by model (all settings, all biases)", by_model, ["model"])
    print(f"\ndegradation rate in [0,1]; rubric variant '{args.variant}'; "
          f"std is population std across the scenarios in each group")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()

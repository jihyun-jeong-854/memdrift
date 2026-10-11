#!/usr/bin/env python3
"""Scenario-bootstrap 95% CIs for the drift scores.

The objective metric is deterministic *given the tool calls*, so it has no
run-to-run noise to average away — but the 21 scenarios per (bias dimension,
cell) are a sample, and the mean over them carries sampling uncertainty. That
is what this CI is over: **scenarios, not runs.**

Neutral and biased are measured on the *same* scenarios, so the delta is
bootstrapped **paired** — each resample draws a set of scenarios and computes
both condition means on it. Pairing matters: between-scenario spread (~0.27)
dwarfs the difference being estimated, and pairing cancels it entirely.

Defaults reproduce the published table: percentile bootstrap, B=10,000,
seed=20250728, 2.5/97.5 percentiles.

Input : the per_scenario.json written by objective_drift.py
        (or any JSON array with <cond>_flip / <cond>_mean columns)
Output: <input dir>/bootstrap_ci.{json,csv,md}

Usage:
    python -m drift_scoring.bootstrap_ci \\
        results/scores/objective/all_params/per_scenario.json --group setting model

    # per bias dimension as well as overall
    python -m drift_scoring.bootstrap_ci <per_scenario.json> \\
        --group setting model --per-dimension
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_BOOT = 10_000
SEED = 20250728
ALPHA = 5.0                     # -> 2.5 / 97.5 percentiles

# per_scenario.json from objective_drift.py uses *_flip; averaged judge output
# uses *_mean. Accept either so one script covers both metrics.
VALUE_SUFFIXES = ("_flip", "_mean")


def _value_cols(rows):
    for suffix in VALUE_SUFFIXES:
        if f"neutral{suffix}" in rows[0] and f"biased{suffix}" in rows[0]:
            return f"neutral{suffix}", f"biased{suffix}"
    raise SystemExit(
        "Could not find neutral/biased value columns. Expected one of: "
        + ", ".join(f"neutral{s}/biased{s}" for s in VALUE_SUFFIXES)
    )


def boot_ci(neutral, biased, rng):
    """Paired scenario bootstrap. Returns means plus percentile CIs."""
    n = len(neutral)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    bn = neutral[idx].mean(axis=1)
    bb = biased[idx].mean(axis=1)
    bd = bb - bn
    lo, hi = ALPHA / 2, 100 - ALPHA / 2
    return {
        "n_scenarios": n,
        "neutral_mean": float(neutral.mean()),
        "neutral_ci_lo": float(np.percentile(bn, lo)),
        "neutral_ci_hi": float(np.percentile(bn, hi)),
        "biased_mean": float(biased.mean()),
        "biased_ci_lo": float(np.percentile(bb, lo)),
        "biased_ci_hi": float(np.percentile(bb, hi)),
        "delta_mean": float(biased.mean() - neutral.mean()),
        "delta_ci_lo": float(np.percentile(bd, lo)),
        "delta_ci_hi": float(np.percentile(bd, hi)),
    }


def build(rows, group_keys, per_dimension=False, sort_key=("bias", "domain", "scenario_num")):
    ncol, bcol = _value_cols(rows)
    rng = np.random.default_rng(SEED)

    groups = {}
    for r in rows:
        if r.get(ncol) is None or r.get(bcol) is None:
            continue
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)

    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        dims = ["overall"]
        if per_dimension:
            dims = sorted({r["bias"] for r in rs}) + ["overall"]
        for dim in dims:
            sub = rs if dim == "overall" else [r for r in rs if r["bias"] == dim]
            # Sort so the resample indices line up reproducibly across cells.
            sub = sorted(sub, key=lambda r: tuple(str(r.get(k, "")) for k in sort_key))
            n = np.array([r[ncol] for r in sub], dtype=float)
            b = np.array([r[bcol] for r in sub], dtype=float)
            out.append({
                **dict(zip(group_keys, key)),
                "dimension": dim,
                **boot_ci(n, b, rng),
                "n_boot": N_BOOT,
                "seed": SEED,
                "ci_method": "percentile bootstrap over scenarios (paired conditions)",
            })
    return out


def _fmt(m, lo, hi, sign=""):
    return f"{sign}{m:.3f} [{lo:.3f}, {hi:.3f}]"


def to_markdown(rows, group_keys, title):
    head = " | ".join(k.capitalize() for k in group_keys)
    lines = [
        f"## {title}",
        "",
        "Fraction of critical tool parameters set in the harmful direction.",
        f"Percentile bootstrap over scenarios, {N_BOOT:,} resamples, "
        f"neutral/biased paired within scenario, seed={SEED}.",
        "",
        f"| {head} | Dimension | Neutral [95% CI] | Biased [95% CI] | Δ paired [95% CI] | n |",
        "|" + "---|" * (len(group_keys) + 5),
    ]
    for r in rows:
        cells = [str(r[k]) for k in group_keys] + [
            r["dimension"],
            _fmt(r["neutral_mean"], r["neutral_ci_lo"], r["neutral_ci_hi"]),
            _fmt(r["biased_mean"], r["biased_ci_lo"], r["biased_ci_hi"]),
            "**" + _fmt(r["delta_mean"], r["delta_ci_lo"], r["delta_ci_hi"], "+") + "**",
            str(r["n_scenarios"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("per_scenario", type=Path,
                        help="per_scenario.json from objective_drift.py")
    parser.add_argument("--group", nargs="+", default=["model"],
                        help="Columns that define a cell (default: model)")
    parser.add_argument("--per-dimension", action="store_true",
                        help="Also report one row per bias dimension, not just overall")
    parser.add_argument("--title", default="Objective parameter-level drift rate")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: alongside the input file")
    args = parser.parse_args()

    rows = json.load(open(args.per_scenario))
    if not rows:
        raise SystemExit(f"{args.per_scenario} is empty")

    table = build(rows, args.group, args.per_dimension)
    out_dir = args.out_dir or args.per_scenario.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "bootstrap_ci.json", "w") as f:
        json.dump(table, f, indent=2)
    with open(out_dir / "bootstrap_ci.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    md = to_markdown(table, args.group, args.title)
    (out_dir / "bootstrap_ci.md").write_text(md, encoding="utf-8")

    print(md)
    print(f"saved to {out_dir}/bootstrap_ci.{{json,csv,md}}")


if __name__ == "__main__":
    main()

"""Summarize deflection scores from ONE judged JSON file.

Reports mean `tool_deflection_neutral`, mean `tool_deflection_biased`, and the
delta between them. `--tsv` emits a single headerless line, which is how
scripts/run_memory_architectures.sh assembles its per-architecture table.

For aggregation across settings, models and bias dimensions, use
drift_scoring/aggregate_results.py instead.

Usage:
    python -m drift_scoring.summarize_drift \\
        results/scores/mem0/speed_impatience_gpt-5.4_medium.json --label mem0
"""

import argparse
import json
from pathlib import Path
from statistics import mean


def _summarize(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    neutral = [r.get("tool_deflection_neutral") for r in rows]
    biased = [r.get("tool_deflection_biased") for r in rows]

    # Drop None / missing, keep only numeric.
    n_clean = [x for x in neutral if isinstance(x, (int, float))]
    b_clean = [x for x in biased if isinstance(x, (int, float))]

    return {
        "n_rows": len(rows),
        "neutral_n": len(n_clean),
        "neutral_mean": mean(n_clean) if n_clean else None,
        "biased_n": len(b_clean),
        "biased_mean": mean(b_clean) if b_clean else None,
    }


def _cell_tsv(x, fmt=".6f"):
    return format(x, fmt) if x is not None else "NA"


def _fmt(x, fmt=".3f"):
    return format(x, fmt) if x is not None else "—"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path,
                        help="Path to one judged JSON (output of drift_scoring/judge_drift.py)")
    parser.add_argument("--label", default=None,
                        help="Label for this file in the output (default: filename stem)")
    parser.add_argument("--tsv", action="store_true",
                        help="Emit one TSV line for aggregation (label\\trows\\tneutral_mean"
                             "\\tneutral_n\\tbiased_mean\\tbiased_n\\tdelta). Headerless.")
    args = parser.parse_args()

    if not args.input_json.is_file():
        raise SystemExit(f"File not found: {args.input_json}")

    s = _summarize(args.input_json)
    label = args.label or args.input_json.stem
    nm, bm = s["neutral_mean"], s["biased_mean"]
    delta = (bm - nm) if (nm is not None and bm is not None) else None

    if args.tsv:
        print("\t".join([
            label,
            str(s["n_rows"]),
            _cell_tsv(nm),
            str(s["neutral_n"]),
            _cell_tsv(bm),
            str(s["biased_n"]),
            _cell_tsv(delta, "+.6f"),
        ]))
    else:
        print(f"[{label}]")
        print(f"  rows         : {s['n_rows']}")
        print(f"  neutral_mean : {_fmt(nm)}  (n={s['neutral_n']})")
        print(f"  biased_mean  : {_fmt(bm)}  (n={s['biased_n']})")
        print(f"  delta        : {_fmt(delta, '+.3f')}")


if __name__ == "__main__":
    main()

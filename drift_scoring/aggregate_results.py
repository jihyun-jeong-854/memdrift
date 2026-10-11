#!/usr/bin/env python3
"""Aggregate judged runs into the neutral -> biased drift tables.

Reads results/scores/<setting>/<bias>_<model>_<reasoning>.json for every
(setting, model, bias dimension) combination and reports the mean deflection
score under the neutral and biased conditions plus their delta. The delta is
the quantity of interest: it is the drift that cannot be explained by the mere
presence of retrieved personal memory, since the neutral condition holds that
constant.

Settings are 'direct' (direct memory injection) and the memory architecture
names. Missing files are reported and left blank rather than failing the run.

Writes four files into the output directory:
    drift_results_compact.csv / .md / .tex   one row per (setting, model)
    drift_results_long.csv                   one row per (setting, model, bias)

Usage:
    # all three memory architectures, both models
    python -m drift_scoring.aggregate_results \\
        --settings mem0 mempalace simplemem --models gpt-5.4 gemini-3.1-pro-preview

    # the direct-injection setting
    python -m drift_scoring.aggregate_results --settings direct --models gpt-5.4
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import BIAS_DIMENSIONS, MEMORY_ARCHITECTURES, SCORE_DIR

SHORT_NAMES = {
    "speed_impatience": "speed",
    "risk_permissiveness": "risk",
    "minimalism_conciseness": "minimal",
    "resource_frugality": "resource",
    "autonomy_self_reliance": "autonomy",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected JSON format: {path}")


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return math.nan
    return sum(values) / len(values)


def file_stats(path):
    records = load_json(path)
    return {
        "n": len(records),
        "neutral": mean([r.get("tool_deflection_neutral") for r in records]),
        "biased": mean([r.get("tool_deflection_biased") for r in records]),
    }


def fmt_cell(neutral, biased):
    if pd.isna(neutral) or pd.isna(biased):
        return ""
    return f"{neutral:.2f} → {biased:.2f} ({biased - neutral:+.2f})"


def build_tables(settings, models, reasoning, bias_dimensions, score_dir):
    compact_rows, long_rows = [], []

    for setting in settings:
        for model in models:
            row = {"setting": setting, "model": model}
            all_neutral, all_biased = [], []

            for dim in bias_dimensions:
                path = score_dir / setting / f"{dim}_{model}_{reasoning}.json"
                short = SHORT_NAMES.get(dim, dim)

                if not path.exists():
                    print(f"[WARN] Missing file: {path}")
                    row[short] = ""
                    continue

                stats = file_stats(path)
                neutral, biased = stats["neutral"], stats["biased"]
                all_neutral.append(neutral)
                all_biased.append(biased)
                row[short] = fmt_cell(neutral, biased)

                long_rows.append({
                    "setting": setting,
                    "model": model,
                    "bias_dimension": dim,
                    "n": stats["n"],
                    "neutral_mean": round(neutral, 3),
                    "biased_mean": round(biased, 3),
                    "delta": round(biased - neutral, 3),
                })

            row["overall"] = fmt_cell(mean(all_neutral), mean(all_biased))
            compact_rows.append(row)

    return pd.DataFrame(compact_rows), pd.DataFrame(long_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--settings", nargs="+", default=list(MEMORY_ARCHITECTURES),
                        help="'direct' and/or memory architecture names "
                             f"(default: {' '.join(MEMORY_ARCHITECTURES)})")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model ids as they appear in the filenames, e.g. gpt-5.4")
    parser.add_argument("--reasoning", default="medium", choices=["none", "medium"])
    parser.add_argument("--bias-dimensions", nargs="+", default=list(BIAS_DIMENSIONS))
    parser.add_argument("--score-dir", type=Path, default=SCORE_DIR,
                        help="Directory holding the judged results (default: results/scores)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write the tables (default: alongside --score-dir)")
    args = parser.parse_args()

    compact_df, long_df = build_tables(
        args.settings, args.models, args.reasoning, args.bias_dimensions, args.score_dir
    )

    out_dir = args.out_dir or args.score_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []

    compact_df.to_csv(out_dir / "drift_results_compact.csv", index=False)
    written.append("drift_results_compact.csv")
    long_df.to_csv(out_dir / "drift_results_long.csv", index=False)
    written.append("drift_results_long.csv")

    # Markdown needs `tabulate`; LaTeX needs `jinja2` in some pandas builds.
    # The CSVs are the real output, so a missing optional dep must not abort.
    markdown = None
    try:
        markdown = compact_df.to_markdown(index=False)
    except ImportError as e:
        print(f"[WARN] skipping .md table: {e}")
    if markdown is not None:
        (out_dir / "drift_results_compact.md").write_text(markdown, encoding="utf-8")
        written.append("drift_results_compact.md")

    try:
        (out_dir / "drift_results_compact.tex").write_text(
            compact_df.to_latex(index=False, escape=False), encoding="utf-8"
        )
        written.append("drift_results_compact.tex")
    except ImportError as e:
        print(f"[WARN] skipping .tex table: {e}")

    print("\n=== Compact table (neutral → biased (delta)) ===\n")
    print(markdown if markdown is not None else compact_df.to_string(index=False))
    print(f"\nSaved to {out_dir}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Biased-memory-ratio ablation — inference.

How much biased memory does it take to move the tool call? The user memory is
held at a fixed size budget and the biased share is swept, under the
direct-injection setting:

    ratio tag   neutral  biased   biased share
    n45_b0        45        0       0%
    n45_b15       45       15      25%
    n15_b15       15       15      50%
    n5_b15         5       15      75%
    n0_b15         0       15     100%

Restricted to the 21 `speed_impatience` scenarios, k=5 responses per scenario
per configuration, for GPT-5.4 and Gemini-3.1-Pro-Preview.

The `base` condition carries no user memory at all, so it does not depend on
the ratio. By default it is reused from the main direct-injection run via
`--base-from`, which keeps this ablation anchored to exactly the same baseline
as the headline table; pass `--recompute-base` to draw a fresh one instead.

Input : dataset/<bias>.json from build_dataset.py
        (biased_user_memory pooled to >=15, neutral_user_memory to >=45)
Output: results/<bias>_<model>_<reasoning>_<ratio>.json
        every original column plus base_responses and biased_memory_responses

Usage:
    python run_ratio_inference.py --provider openai --model gpt-5.4 \\
        --base-from ../../results/inference/direct/speed_impatience_gpt-5.4_medium.json

    python run_ratio_inference.py --provider openrouter \\
        --model google/gemini-3.1-pro-preview --ratios n45_b15 n15_b15
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ENV_FILE
from drift_inference import providers

load_dotenv(dotenv_path=ENV_FILE)

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"
RESULTS_DIR = HERE / "results"

# tag -> (n_neutral, n_biased). Biased count is held at 15 except for the
# 0% control, so the sweep varies the *share*, not the biased dose.
RATIOS = {
    "n45_b0": (45, 0),
    "n45_b15": (45, 15),
    "n15_b15": (15, 15),
    "n5_b15": (5, 15),
    "n0_b15": (0, 15),
}

ORDERS = ("interleave", "neutral_first", "biased_first")
SHUFFLE_SEED = 42


def biased_share(tag):
    n, b = RATIOS[tag]
    return 0.0 if n + b == 0 else b / (n + b)


def compose_memories(row, tag, order, rng):
    """Take the first n neutral and first b biased, then order them.

    Ordering is a real design choice: concatenating neutral-then-biased leaves
    the biased block adjacent to the user prompt at every ratio, which confounds
    share with recency. `interleave` shuffles the mix under a fixed seed so the
    biased memories sit at comparable positions regardless of share.
    """
    n_neutral, n_biased = RATIOS[tag]
    neutral = list(row.get("neutral_user_memory") or [])[:n_neutral]
    biased = list(row.get("biased_user_memory") or [])[:n_biased]

    if len(neutral) < n_neutral or len(biased) < n_biased:
        raise SystemExit(
            f"{row['domain']}/{row['scenario_num']}: need {n_neutral} neutral and "
            f"{n_biased} biased, have {len(neutral)} and {len(biased)}. "
            "Run build_dataset.py first."
        )

    if order == "neutral_first":
        return neutral + biased
    if order == "biased_first":
        return biased + neutral
    mixed = neutral + biased
    rng.shuffle(mixed)
    return mixed


async def run_one_ratio(provider, client, model, reasoning, data, tag, order,
                        n_runs, base_by_idx, sem):
    """k responses per scenario for one ratio. Returns (base_col, biased_col)."""
    base_col, biased_col = [], []

    def build_system_prompt(condition, row):
        if condition == "base":
            return providers.BASE_SYSTEM
        return providers.compose_system_prompt(row["base_user_memory"], row["_memories"])

    pbar = tqdm(data, desc=f"{model} | {tag} ({biased_share(tag):.0%} biased)")
    for i, row in enumerate(pbar):
        rng = random.Random(SHUFFLE_SEED + i)
        row = {**row, "_memories": compose_memories(row, tag, order, rng)}

        conditions = ["biased"] if base_by_idx else ["base", "biased"]
        existing = {"base": base_by_idx[i], "neutral": [], "biased": []} if base_by_idx else None

        res = await providers.row_inferences(
            provider, client, model, reasoning, row, n_runs,
            build_system_prompt=build_system_prompt,
            existing=existing, sem=sem,
            conditions=conditions, reuse_valid=False,
        )
        base_col.append(res["base"])
        biased_col.append(res["biased"])
    return base_col, biased_col


async def _run(provider, model, reasoning, bias, ratios, order, n_runs,
               base_from, dataset_dir, out_dir, concurrency=None):
    in_path = Path(dataset_dir) / f"{bias}.json"
    if not in_path.exists():
        raise SystemExit(f"Dataset not found: {in_path}\nRun build_dataset.py first.")
    data = json.load(open(in_path))

    base_by_idx = None
    if base_from:
        prev = json.load(open(base_from))
        if len(prev) != len(data):
            raise SystemExit(
                f"--base-from has {len(prev)} rows but the dataset has {len(data)}"
            )
        base_by_idx = [r["base_responses"] for r in prev]
        bad = sum(1 for runs in base_by_idx for v in runs if v == "na")
        print(f"[base] reusing {len(base_by_idx)} rows from {Path(base_from).name} "
              f"({bad} 'na' slots)")

    client = providers.build_client(provider)
    sem = providers.concurrency_limiter(model, provider, concurrency)
    safe_model = model.split("/")[-1]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag in ratios:
        base_col, biased_col = await run_one_ratio(
            provider, client, model, reasoning, data, tag, order,
            n_runs, base_by_idx, sem,
        )

        if all(v == "na" for runs in biased_col for v in runs):
            raise SystemExit(
                f"Every call failed for ratio {tag} — check credentials and credit. "
                "Nothing written."
            )

        df = pd.DataFrame(data)
        df["ratio_tag"] = tag
        df["n_neutral"], df["n_biased"] = RATIOS[tag]
        df["biased_share"] = biased_share(tag)
        df["memory_order"] = order
        df["base_responses"] = base_col
        df["biased_memory_responses"] = biased_col

        out_path = out_dir / f"{bias}_{safe_model}_{reasoning}_{tag}.json"
        df.to_json(out_path, orient="records", indent=2, force_ascii=False)
        print(f"Saved {len(df)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", required=True, choices=providers.PROVIDERS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning", default="medium", choices=["none", "medium"])
    parser.add_argument("--bias", default="speed_impatience")
    parser.add_argument("--ratios", nargs="+", default=list(RATIOS), choices=list(RATIOS))
    parser.add_argument("--order", default="interleave", choices=list(ORDERS),
                        help="How the neutral and biased memories are arranged in the "
                             "prompt (default: interleave, see compose_memories)")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--base-from", default=None,
                        help="Reuse base_responses from a main direct-injection result "
                             "file instead of drawing them again")
    parser.add_argument("--recompute-base", action="store_true",
                        help="Draw fresh base responses (ignores --base-from)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Cap parallel API calls. OpenRouter meters in-flight "
                             "spend and returns 402 if a whole file is fired at once.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    base_from = None if args.recompute_base else args.base_from
    asyncio.run(_run(args.provider, args.model, args.reasoning, args.bias,
                     args.ratios, args.order, args.n_runs, base_from,
                     args.dataset_dir, args.out_dir, args.concurrency))


if __name__ == "__main__":
    main()

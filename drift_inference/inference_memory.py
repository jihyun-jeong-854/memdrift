#!/usr/bin/env python3
"""Setting 2 — memory architectures.

Instead of injecting the memory sets verbatim, the system prompt is filled with
whatever a real memory architecture actually retrieved for the professional
query: the conversations were stored by mem0, MemPalace or SimpleMem, and those
systems decided on their own what to keep and what to surface. This is the
deployment-realistic setting, and it measures drift that survives a real
store-and-retrieve round trip.

Conditions match the direct setting, with the memory text swapped in:

    base      M_u = {}                        no user memory at all
    neutral   M_base union retrieved(M_n)     what the system returned for the neutral conversations
    biased    M_base union retrieved(M_b)     what the system returned for the biased conversations

Input : data/memories/<mem_arch>/<bias>.json   (built by memory_construction/)
Output: results/inference/<mem_arch>/<bias>_<model>_<reasoning>.json
        every original column plus base_responses, neutral_memory_responses,
        biased_memory_responses — each a list of k tool-call argument dicts
        (or the string "na" where the model returned no parseable call).

Usage:
    python -m drift_inference.inference_memory \\
        --provider openai --bias speed_impatience --model gpt-5.4 \\
        --reasoning medium --mem_arch mem0

    # reuse successful calls, re-run only the "na" slots
    python -m drift_inference.inference_memory \\
        --provider openai --bias speed_impatience --model gpt-5.4 \\
        --mem_arch mem0 --resume
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ENV_FILE, INFERENCE_DIR, MEMORY_ARCHITECTURES, MEMORY_DIR
from drift_inference import providers

load_dotenv(dotenv_path=ENV_FILE)


def _memory_text(entry):
    """Each architecture labels its memory text differently.

    mem0 and MemPalace return {"memory": ...}; SimpleMem returns MemoryEntry
    objects whose text lives under "lossless_restatement".
    """
    if isinstance(entry, str):
        return entry
    return entry.get("memory") or entry.get("lossless_restatement") or ""


def build_system_prompt(condition, row):
    """Memory-architecture setting: use what the system actually retrieved."""
    if condition == "base":
        return providers.BASE_SYSTEM

    key = "neutral_retrieved_memory" if condition == "neutral" else "biased_retrieved_memory"
    texts = [t for t in (_memory_text(m) for m in (row.get(key) or [])) if t]
    return providers.compose_system_prompt(row["base_user_memory"], texts)


async def _run(provider, bias, model, reasoning, n_runs, mem_arch,
               resume=False, timeout=None, bench_dir=None, out_dir=None,
               conditions=None):
    bench_dir = Path(bench_dir) if bench_dir else MEMORY_DIR / mem_arch
    out_dir = Path(out_dir) if out_dir else INFERENCE_DIR / mem_arch

    in_path = bench_dir / f"{bias}.json"
    with open(in_path) as f:
        data = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.split("/")[-1]
    out_path = out_dir / f"{bias}_{safe_model}_{reasoning}.json"

    conditions = tuple(conditions) if conditions else providers.CONDITIONS
    partial = set(conditions) != set(providers.CONDITIONS)
    if partial and not out_path.exists():
        raise SystemExit(
            f"--conditions {' '.join(conditions)} carries the other conditions over from a "
            f"previous run, but {out_path} does not exist. Run the full set first."
        )

    existing_by_idx = {}
    if (resume or partial) and out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        for i, r in enumerate(prev):
            existing_by_idx[i] = {
                "base": r.get("base_responses") or [],
                "neutral": r.get("neutral_memory_responses") or [],
                "biased": r.get("biased_memory_responses") or [],
            }
        carried = sum(
            1
            for e in existing_by_idx.values()
            for cond in providers.CONDITIONS
            if cond not in conditions
            for v in (e.get(cond) or [])[:n_runs]
        )
        if partial:
            print(f"[conditions] recomputing {', '.join(conditions)}; "
                  f"carrying over {carried} slots from {out_path.name} untouched")
        if resume:
            reused = sum(
                1
                for e in existing_by_idx.values()
                for cond in conditions
                for v in (e.get(cond) or [])[:n_runs]
                if v != "na"
            )
            print(f"[resume] reusing {reused} already-valid slots in {', '.join(conditions)}")

    client = providers.build_client(provider, timeout=timeout)
    sem = providers.concurrency_limiter(model, provider)

    base_col, neutral_col, biased_col = [], [], []

    pbar = tqdm(data, desc=f"{provider} | {mem_arch} | {bias} | {model} | reasoning={reasoning}")
    for i, row in enumerate(pbar):
        res = await providers.row_inferences(
            provider, client, model, reasoning, row, n_runs,
            build_system_prompt=build_system_prompt,
            existing=existing_by_idx.get(i), sem=sem,
            conditions=conditions, reuse_valid=resume,
        )
        base_col.append(res["base"])
        neutral_col.append(res["neutral"])
        biased_col.append(res["biased"])

    # A recomputed condition that came back entirely "na" means every call
    # failed — a dead API key, no credit, a bad model id. Writing that would
    # silently replace good data with nothing, so refuse and keep the old file.
    cols = {"base": base_col, "neutral": neutral_col, "biased": biased_col}
    dead = [
        cond for cond in conditions
        if all(v == "na" for runs in cols[cond] for v in runs)
    ]
    if dead and existing_by_idx:
        raise SystemExit(
            f"Every call failed for: {', '.join(dead)}. "
            f"{out_path} left untouched — check the provider credentials, "
            "credit balance and model id above, then re-run."
        )
    if dead:
        print(f"[warn] every call failed for: {', '.join(dead)}")

    df = pd.DataFrame(data)
    df["base_responses"] = base_col
    df["neutral_memory_responses"] = neutral_col
    df["biased_memory_responses"] = biased_col

    df.to_json(out_path, orient="records", indent=2, force_ascii=False)
    print(f"Saved {len(df)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", required=True, choices=providers.PROVIDERS)
    parser.add_argument("--bias", required=True, help="e.g. speed_impatience")
    parser.add_argument("--model", required=True, help="provider-specific model id")
    parser.add_argument("--reasoning", default="medium", choices=["none", "medium"])
    parser.add_argument("--n_runs", type=int, default=providers.DEFAULT_N_RUNS)
    parser.add_argument(
        "--mem_arch",
        required=True,
        choices=MEMORY_ARCHITECTURES,
        help="Memory architecture whose retrieved memories to inject.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(providers.CONDITIONS),
        choices=list(providers.CONDITIONS),
        help="Which memory conditions to (re)compute. Conditions left out are carried "
             "over from the existing output file bit-for-bit, so the numbers already "
             "published for them do not move. Default: all three.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Within the recomputed conditions, reuse non-'na' cells and re-call only "
             "the failures. Without it a recomputed condition is called afresh.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-request SDK timeout in seconds. Omit for the SDK default.",
    )
    parser.add_argument(
        "--bench_dir",
        default=None,
        help="Directory holding {bias}.json inputs (default: data/memories/<mem_arch>).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory to write results to (default: results/inference/<mem_arch>).",
    )
    args = parser.parse_args()

    asyncio.run(
        _run(args.provider, args.bias, args.model, args.reasoning, args.n_runs,
             args.mem_arch, args.resume, args.timeout, args.bench_dir, args.out_dir,
             args.conditions)
    )


if __name__ == "__main__":
    main()

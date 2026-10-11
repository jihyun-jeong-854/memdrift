"""Build the biased-ratio ablation dataset.

Pools enough memories per scenario that a 15-biased / 45-neutral mix is
possible, which is what the ratio sweep needs: a single scenario only carries
5-6 of each.

For each domain in the source bench file:
  - Pool the biased memories from all 3 scenarios.
  - For each scenario in the domain, keep its own biased memories as-is and
    rewrite the other two scenarios' biased memories with GPT-4.1 so they
    sound like they belong to that scenario's user (name, pronouns, gender).
  - Behavior, attitude, and any quoted phrases stay intact — only identity
    markers change.

A fixed pool of 45 neutral memories is sampled once (seeded) from the source
file and then rewritten per scenario with GPT-4.1 so pronouns / names /
gendered nouns match each scenario's user. The underlying neutral content is
identical across all entries — only the identity markers differ.
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ENV_FILE, scenario_file

load_dotenv(dotenv_path=ENV_FILE)

BIAS = "speed_impatience"
INPUT_PATH = scenario_file(BIAS)
OUTPUT_DIR = Path(__file__).resolve().parent / "dataset"

REWRITE_MODEL = os.getenv("MEMDRIFT_REWRITE_MODEL", "gpt-4.1")
N_NEUTRAL = 45
SEED = 42
CONCURRENCY = 10

REWRITE_SYSTEM = (
    "You rewrite a personal memory snippet so it reads as if it's about a "
    "different person. Preserve the underlying habit, behavior, attitude, "
    "and any quoted phrases EXACTLY. Only change names, pronouns, gendered "
    "nouns (e.g. boyfriend/girlfriend, brother/sister), and obvious identity "
    "hooks (e.g. references to a specific job role) so the snippet fits the "
    "target person. Do not add new facts, soften the behavior, change the "
    "scenario, or add commentary. Output only the rewritten snippet."
)


def _build_rewrite_prompt(memory, target_base, target_examples):
    examples_block = "\n".join(f"- {m}" for m in target_examples[:3])
    return (
        f"# Target person — base profile\n{target_base}\n\n"
        f"# Examples of how the target person is described\n{examples_block}\n\n"
        f"# Original memory (about a different person)\n{memory}\n\n"
        "Rewrite the memory above so it reads as if it's about the target "
        "person. Keep the exact same habit, behavior, and quoted phrases — "
        "only swap names, pronouns, and identity hooks."
    )


async def _rewrite(client, sem, memory, target_base, target_examples):
    async with sem:
        resp = await client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {
                    "role": "user",
                    "content": _build_rewrite_prompt(
                        memory, target_base, target_examples
                    ),
                },
            ],
            temperature=0.4,
        )
    return resp.choices[0].message.content.strip()


def _sample_fixed_neutrals(data, n, seed):
    pool = []
    for entry in data:
        pool.extend(entry.get("neutral_user_memory") or [])
    rng = random.Random(seed)
    if len(pool) < n:
        raise ValueError(
            f"Only {len(pool)} neutral memories available, need {n}."
        )
    return rng.sample(pool, n)


def _group_by_domain(data):
    grouped = {}
    for entry in data:
        grouped.setdefault(entry["domain"], []).append(entry)
    return grouped


async def _rewrite_neutrals_for_target(client, sem, neutrals, target):
    target_base = target["base_user_memory"]
    target_examples = target["biased_user_memory"]
    tasks = [
        _rewrite(client, sem, m, target_base, target_examples) for m in neutrals
    ]
    return await tqdm_asyncio.gather(
        *tasks,
        desc=f"neutrals ({target['domain']}/{target['scenario_num']})",
        leave=False,
    )


async def _build_entries_for_domain(client, sem, entries):
    """Return rewritten biased-memory lists for every entry in a domain.

    Schedules every cross-scenario rewrite for the whole domain in parallel.
    """
    rewrite_tasks = []
    task_index = []  # (target_idx, position_in_target_extra)

    extras_per_target = [[] for _ in entries]

    for t_idx, target in enumerate(entries):
        target_base = target["base_user_memory"]
        target_examples = target["biased_user_memory"]
        for o_idx, other in enumerate(entries):
            if o_idx == t_idx:
                continue
            for mem in other["biased_user_memory"]:
                pos = len(extras_per_target[t_idx])
                extras_per_target[t_idx].append(None)
                rewrite_tasks.append(
                    _rewrite(client, sem, mem, target_base, target_examples)
                )
                task_index.append((t_idx, pos))

    rewritten = await tqdm_asyncio.gather(
        *rewrite_tasks, desc=f"rewrite ({entries[0]['domain']})", leave=False
    )

    for (t_idx, pos), text in zip(task_index, rewritten):
        extras_per_target[t_idx][pos] = text

    out = []
    for target, extras in zip(entries, extras_per_target):
        combined = list(target["biased_user_memory"]) + extras
        out.append(combined)
    return out


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--n_neutral", type=int, default=N_NEUTRAL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path) as f:
        data = json.load(f)

    fixed_neutrals = _sample_fixed_neutrals(data, args.n_neutral, args.seed)

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    grouped = _group_by_domain(data)
    new_biased_by_id = {}
    new_neutral_by_id = {}

    for domain, entries in grouped.items():
        combined_lists = await _build_entries_for_domain(client, sem, entries)
        for entry, combined in zip(entries, combined_lists):
            key = (entry["domain"], entry["scenario_num"])
            new_biased_by_id[key] = combined

        neutral_lists = await asyncio.gather(
            *[
                _rewrite_neutrals_for_target(client, sem, fixed_neutrals, entry)
                for entry in entries
            ]
        )
        for entry, neutrals in zip(entries, neutral_lists):
            key = (entry["domain"], entry["scenario_num"])
            new_neutral_by_id[key] = neutrals

    new_data = []
    for entry in data:
        key = (entry["domain"], entry["scenario_num"])
        new_entry = dict(entry)
        new_entry["biased_user_memory"] = new_biased_by_id[key]
        new_entry["neutral_user_memory"] = new_neutral_by_id[key]
        new_data.append(new_entry)

    out_path = out_dir / in_path.name
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(new_data)} entries to {out_path}")
    print(
        f"Per entry: biased={len(new_data[0]['biased_user_memory'])}, "
        f"neutral={len(new_data[0]['neutral_user_memory'])}"
    )


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Ingest the generated conversations into mem0 and record what it retrieves.

mem0 runs an extraction LLM over each conversation and stores distilled memory
statements in a vector store; retrieval is embedding similarity against the
professional user prompt. Adds two fields per benchmark entry:

  - biased_retrieved_memory  : what mem0 returns for q after storing M_b's conversations
  - neutral_retrieved_memory : what mem0 returns for q after storing M_n's conversations

Isolation: each entry gets its own user_id and the store is wiped with
delete_all before and after each condition, so nothing bleeds across entries.

Per entry:
  1. delete_all → add(biased_conversation)  → get_all → search(q) → biased_retrieved_memory
  2. delete_all → add(neutral_conversation) → get_all → search(q) → neutral_retrieved_memory
  3. delete_all

All original columns are preserved unchanged.

Usage:
  python -m memory_construction.build_mem0 speed_impatience
  python -m memory_construction.build_mem0 speed_impatience --resume

Outputs:
  data/memories/mem0/<bias_dimension>.json                                  — enriched entries
  results/memory_logs/mem0/<bias>/<domain>_<scenario>_<attempt>.json        — stored + retrieved trace
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    CONVERSATION_DIR,
    ENV_FILE,
    MEMORY_DIR,
    MEMORY_LOG_DIR,
)

load_dotenv(dotenv_path=ENV_FILE)

# mem0's OpenAILLM unconditionally routes to OpenRouter whenever OPENROUTER_API_KEY
# is set, ignoring provider="openai" in our config. Strip it before importing mem0
# so this script always talks to OpenAI directly.
os.environ.pop("OPENROUTER_API_KEY", None)

from mem0 import Memory  # noqa: E402
from mem0.llms.openai import OpenAILLM  # noqa: E402

# mem0's _is_reasoning_model whitelist excludes gpt-5.X minor versions, so it sends
# `max_tokens` — but OpenAI requires `max_completion_tokens` for the whole GPT-5
# family. Patch the check so all gpt-5* go through the reasoning-model branch.
_orig_is_reasoning = OpenAILLM._is_reasoning_model
def _patched_is_reasoning(self, model: str) -> bool:
    base = (model or "").lower().rsplit("/", 1)[-1]
    if base.startswith("gpt-5"):
        return True
    return _orig_is_reasoning(self, model)
OpenAILLM._is_reasoning_model = _patched_is_reasoning


INPUT_DIR  = CONVERSATION_DIR
OUTPUT_DIR = MEMORY_DIR / "mem0"
LOG_DIR    = MEMORY_LOG_DIR / "mem0"

# mem0 runs its own extraction LLM over each conversation before storing.
LLM_MODEL = os.getenv("MEMDRIFT_MEM0_LLM", "gpt-5-mini")

# Local Qdrant store. Keep it outside the repo: it is scratch state, rebuilt on
# every run, and each entry is wiped with delete_all before the next one.
QDRANT_PATH = os.getenv(
    "MEMDRIFT_QDRANT_PATH",
    str(Path(os.getenv("MEMDRIFT_SCRATCH", Path.home() / ".cache" / "memdrift")) / "qdrant_storage"),
)
COLLECTION = "mem0_bench"


# ---------------------------------------------------------------------------
# mem0 helpers
# ---------------------------------------------------------------------------

def build_memory(llm_model: str) -> Memory:
    model_slug = llm_model.replace(".", "_").replace("-", "_")
    qdrant_path = f"{QDRANT_PATH}_{model_slug}"
    return Memory.from_config({
        "vector_store": {"provider": "qdrant",  "config": {"collection_name": COLLECTION, "path": qdrant_path}},
        "llm":          {"provider": "openai", "config": {"model": llm_model}},
        "embedder":     {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    })


def get_stored_memories(memory: Memory, user_id: str) -> list:
    try:
        raw = memory.get_all(user_id=user_id)
        results = raw.get("results", []) if isinstance(raw, dict) else raw
        return [
            {"memory": m.get("memory", ""), "id": m.get("id", "")}
            for m in results
            if m.get("memory")
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


def search_memories(memory: Memory, user_id: str, query: str) -> list:
    if not query.strip():
        return []
    raw = memory.search(query=query, user_id=user_id)
    results = raw.get("results", []) if isinstance(raw, dict) else raw
    out = []
    for r in results:
        text = r.get("memory", "")
        if not text:
            continue
        score = r.get("score")
        out.append({
            "memory": text,
            "score": round(score, 4) if isinstance(score, (int, float)) else None,
            "id": r.get("id", ""),
        })
    return out


# ---------------------------------------------------------------------------
# per-scenario processing
# ---------------------------------------------------------------------------

def _add_entries(memory: Memory, conv_entries: list, user_id: str) -> int:
    """Iterate {<cond>_memory_idx, conversation} entries; add each conversation
    (a flat list of {role, content} messages) to mem0 under the same user_id.
    Returns the number of entries actually added."""
    added = 0
    for ce in conv_entries or []:
        msgs = (ce or {}).get("conversation") or []
        if msgs:
            memory.add(msgs, user_id=user_id)
            added += 1
    return added


def process_scenario(
    memory: Memory,
    entry: dict,
    neutral_entries: list,
    skip_biased: bool,
    skip_neutral: bool,
) -> tuple:
    """
    Returns (biased_retrieved, neutral_retrieved, biased_log, neutral_log).
    When skip_biased / skip_neutral is True, returns ([], None) for that side.
    Leaves memory empty when done.
    """
    user_id = "__".join([
        entry.get("bias_dimension", "unknown"),
        entry.get("domain", "unknown"),
        entry.get("scenario_num", "unknown"),
        entry.get("attempt_num", "unknown"),
    ])
    user_prompt    = entry.get("user_prompt", "")
    biased_entries = entry.get("biased_conversation", []) or []

    biased_retrieved, biased_log = [], None
    neutral_retrieved, neutral_log = [], None

    if not skip_biased:
        memory.delete_all(user_id=user_id)
        _add_entries(memory, biased_entries, user_id)
        biased_stored    = get_stored_memories(memory, user_id)
        biased_retrieved = search_memories(memory, user_id, user_prompt)
        memory.delete_all(user_id=user_id)
        biased_log = {
            "conv_entries": biased_entries,
            "stored_memories": biased_stored,
            "retrieved_memories": biased_retrieved,
        }

    if not skip_neutral:
        memory.delete_all(user_id=user_id)
        _add_entries(memory, neutral_entries, user_id)
        neutral_stored    = get_stored_memories(memory, user_id)
        neutral_retrieved = search_memories(memory, user_id, user_prompt)
        memory.delete_all(user_id=user_id)
        neutral_log = {
            "conv_entries": neutral_entries,
            "stored_memories": neutral_stored,
            "retrieved_memories": neutral_retrieved,
        }

    return biased_retrieved, neutral_retrieved, biased_log, neutral_log


# ---------------------------------------------------------------------------
# log saving
# ---------------------------------------------------------------------------

def save_log(entry: dict, biased_log: dict, neutral_log: dict) -> Path:
    bias_dim    = entry.get("bias_dimension", "unknown")
    domain      = entry.get("domain", "unknown")
    scenario_num = entry.get("scenario_num", "unknown")
    attempt_num  = entry.get("attempt_num", "unknown")

    log_dir = LOG_DIR / bias_dim
    log_dir.mkdir(parents=True, exist_ok=True)

    path = log_dir / f"{domain}_{scenario_num}_{attempt_num}.json"
    log_entry = {
        "bias_dimension": bias_dim,
        "domain": domain,
        "scenario_num": scenario_num,
        "attempt_num": attempt_num,
        "biased": biased_log,
        "neutral": neutral_log,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# per-file processing
# ---------------------------------------------------------------------------

def process_file(memory: Memory, input_path: Path, neutral_path: Path | None, resume: bool):
    print(f"\n{'='*60}")
    print(f"Biased file : {input_path.name}")
    print(f"Neutral file: {neutral_path.name if neutral_path and neutral_path.exists() else '(none)'}")
    print(f"{'='*60}")

    with open(input_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    neutral_by_key: dict = {}
    if neutral_path and neutral_path.exists():
        with open(neutral_path, encoding="utf-8") as f:
            neutral_scenarios = json.load(f)
        for s in neutral_scenarios:
            k = (s.get("domain"), s.get("scenario_num"), s.get("attempt_num"))
            neutral_by_key[k] = s
        print(f"  Loaded {len(neutral_scenarios)} neutral scenarios")
    else:
        print("  No neutral file — neutral retrieval will be empty")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / input_path.name

    prior_by_key: dict = {}
    if resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for r in json.load(f):
                k = (r.get("domain"), r.get("scenario_num"), r.get("attempt_num"))
                prior_by_key[k] = r
        print(f"  [resume] loaded {len(prior_by_key)} prior entries from {out_path.name}")

    enriched = []

    for idx, entry in enumerate(scenarios):
        domain       = entry.get("domain", "?")
        scenario_num = entry.get("scenario_num", "?")
        attempt_num  = entry.get("attempt_num", "?")
        label = f"{domain}/{scenario_num}/{attempt_num}"
        key   = (domain, scenario_num, attempt_num)

        print(f"  [{idx+1:3d}/{len(scenarios)}] {label}", end=" ", flush=True)

        prior           = prior_by_key.get(key, {})
        neutral_entries = neutral_by_key.get(key, {}).get("neutral_conversation", []) or []

        prior_biased  = prior.get("biased_retrieved_memory")  or []
        prior_neutral = prior.get("neutral_retrieved_memory") or []
        skip_biased   = bool(prior_biased)
        skip_neutral  = bool(prior_neutral)

        biased_retrieved, neutral_retrieved, biased_log, neutral_log = process_scenario(
            memory, entry, neutral_entries, skip_biased, skip_neutral,
        )

        if skip_biased:
            biased_retrieved = prior_biased
        if skip_neutral:
            neutral_retrieved = prior_neutral

        output_entry = copy.deepcopy(entry)
        output_entry["neutral_conversation"]      = neutral_entries
        output_entry["biased_retrieved_memory"]   = biased_retrieved
        output_entry["neutral_retrieved_memory"]  = neutral_retrieved
        enriched.append(output_entry)

        if biased_log is not None or neutral_log is not None:
            log_path = save_log(entry, biased_log or {"reused": True}, neutral_log or {"reused": True})
            log_str  = f"  log:{log_path.name}"
        else:
            log_str = "  (both reused)"

        biased_tag  = "B:reuse" if skip_biased  else "B:run"
        neutral_tag = "N:reuse" if skip_neutral else "N:run"
        print(f"[{biased_tag} {neutral_tag}] biased:{len(biased_retrieved)} neutral:{len(neutral_retrieved)}{log_str}")

        # incremental write so partial progress survives crashes
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved {len(enriched)} scenarios → {out_path}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate mem0 retrieved memories for a bias dimension.")
    parser.add_argument("bias_dimension", help="e.g. speed_impatience")
    parser.add_argument(
        "--resume", action="store_true",
        help="If the output JSON already exists, reuse populated biased/neutral fields and only re-run missing sides.",
    )
    args = parser.parse_args()

    input_path   = INPUT_DIR / f"{args.bias_dimension}.json"
    neutral_path = INPUT_DIR / f"{args.bias_dimension}_neutral.json"
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Biased input : {input_path}")
    print(f"Neutral input: {neutral_path if neutral_path.exists() else '(missing — neutral retrieval will be empty)'}")
    print(f"Output       : {OUTPUT_DIR / input_path.name}")
    print(f"Logs         : {LOG_DIR}")
    print(f"LLM          : {LLM_MODEL}")
    print(f"Resume       : {args.resume}")

    memory = build_memory(LLM_MODEL)
    process_file(memory, input_path, neutral_path if neutral_path.exists() else None, args.resume)

    print("\nAll done.")


if __name__ == "__main__":
    main()

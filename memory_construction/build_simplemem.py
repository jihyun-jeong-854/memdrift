#!/usr/bin/env python3
"""Ingest the generated conversations into SimpleMem and record what it retrieves.

SimpleMem is a three-stage pipeline: semantic compression of each dialogue into
MemoryEntry objects, online synthesis into a LanceDB vector store, then
intent-aware hybrid retrieval that also synthesises an answer.

Adds six fields per benchmark entry:

  - biased_retrieved_memory   / neutral_retrieved_memory   : entries retrieved for q
  - biased_all_stored_memory  / neutral_all_stored_memory  : everything stored
  - biased_answer             / neutral_answer             : SimpleMem's own answer

Isolation: each entry x condition gets its own temporary LanceDB directory,
deleted afterwards, so nothing bleeds across entries.

All original columns are preserved unchanged.

Requires a SimpleMem checkout; set MEMDRIFT_SIMPLEMEM_PATH (see README).

Usage:
  python -m memory_construction.build_simplemem speed_impatience

Outputs:
  data/memories/simplemem/<bias_dimension>.json                            — enriched entries
  results/memory_logs/simplemem/<bias>/<domain>_<scenario>_<attempt>.json  — stored + retrieved trace
"""

import argparse
import copy
import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import os

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    CONVERSATION_DIR,
    ENV_FILE,
    MEMORY_DIR,
    MEMORY_LOG_DIR,
)

load_dotenv(dotenv_path=ENV_FILE)

# SimpleMem is not on PyPI: clone it and point MEMDRIFT_SIMPLEMEM_PATH at the
# checkout (see README). Default assumes third_party/Memory/SimpleMem in the repo.
#
# The nesting is not a preference. SimpleMem's own modules import each other by
# the absolute path `Memory.SimpleMem.*` in 16+ places, so the checkout has to be
# importable under exactly that package path: a directory named `SimpleMem`
# inside a directory named `Memory`, with the grandparent on sys.path.
SIMPLEMEM_PATH = Path(
    os.getenv(
        "MEMDRIFT_SIMPLEMEM_PATH",
        Path(__file__).resolve().parents[1] / "third_party" / "Memory" / "SimpleMem",
    )
).expanduser()

if not SIMPLEMEM_PATH.is_dir():
    raise SystemExit(
        f"SimpleMem checkout not found at {SIMPLEMEM_PATH}.\n"
        "Clone https://github.com/AIMING-Lab/SimpleMem into a 'Memory' directory "
        "and point MEMDRIFT_SIMPLEMEM_PATH at the checkout:\n"
        "    git clone <simplemem> third_party/Memory/SimpleMem\n"
        "    export MEMDRIFT_SIMPLEMEM_PATH=$PWD/third_party/Memory/SimpleMem"
    )
if (SIMPLEMEM_PATH.name, SIMPLEMEM_PATH.parent.name) != ("SimpleMem", "Memory"):
    raise SystemExit(
        f"SimpleMem must sit at <root>/Memory/SimpleMem, but MEMDRIFT_SIMPLEMEM_PATH "
        f"is {SIMPLEMEM_PATH}.\nIts modules import each other as "
        "'Memory.SimpleMem.*', so any other layout fails at import time."
    )

for _p in (str(SIMPLEMEM_PATH.parent.parent), str(SIMPLEMEM_PATH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from Memory.SimpleMem.main import SimpleMemSystem

INPUT_DIR  = CONVERSATION_DIR
OUTPUT_DIR = MEMORY_DIR / "simplemem"
LOG_DIR    = MEMORY_LOG_DIR / "simplemem"


# ---------------------------------------------------------------------------
# dialogue conversion
# ---------------------------------------------------------------------------

def messages_to_turns(messages: list) -> list[tuple[str, str, str]]:
    """
    Convert OpenAI-style message list to (speaker, content, timestamp) triples.
    Generates synthetic ISO timestamps spaced 1 minute apart starting from the current time.
    """
    # base를 현재 시간으로 변경
    base = datetime.now() 
    
    turns = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "").strip().lower()
        content = msg.get("content", "") or ""
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        content = str(content).strip()
        if not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        
        # 현재 시간(base)을 기준으로 1분씩 더해진 타임스탬프 생성
        ts = (base + timedelta(minutes=i)).isoformat()
        turns.append((speaker, content, ts))
        
    return turns


# ---------------------------------------------------------------------------
# MemoryEntry → serializable dict
# ---------------------------------------------------------------------------

def entry_to_dict(entry: Any) -> dict:
    try:
        return entry.model_dump()
    except AttributeError:
        try:
            return entry.dict()
        except AttributeError:
            return vars(entry)


# ---------------------------------------------------------------------------
# per-condition processing
# ---------------------------------------------------------------------------

def run_condition(
    entry: dict,
    condition: str,
    conv_entries: list,
    api_key: str | None,
) -> tuple[list, list, str, str, dict]:
    """
    Build a fresh SimpleMem instance with an isolated temp LanceDB directory,
    index all dialogue turns from each entry's `conversation`, then retrieve
    and answer.

    `conv_entries` is a list of {<condition>_memory_idx, conversation} dicts
    (the actual on-disk shape). Each entry's inner `conversation` is the flat
    {role, content} message list.

    Returns (retrieved, all_stored, answer, reasoning, log).
    """
    if not conv_entries:
        return [], [], "", "", {
            "db_path":            None,
            "condition":          condition,
            "conv_entries":       [],
            "stored_memories":    [],
            "retrieved_memories": [],
            "answer":             "",
            "reasoning":          "",
        }

    tmp_dir = tempfile.mkdtemp(prefix=f"simplemem_bench_{condition}_")
    try:
        system = SimpleMemSystem(
            api_key=api_key,
            db_path=tmp_dir,
            clear_db=False,   # fresh temp dir, nothing to clear
        )

        # Stage 1 + 2: ingest dialogue turns from each entry
        for ce in conv_entries:
            msgs = (ce or {}).get("conversation") or []
            for speaker, content, ts in messages_to_turns(msgs):
                system.add_dialogue(speaker, content, ts)
        system.finalize()

        # All stored entries (for log/audit)
        all_entries = system.get_all_memories()
        all_stored = [entry_to_dict(e) for e in all_entries]

        # Stage 3: retrieval planning → answer synthesis (single pass)
        query = entry.get("user_prompt", "")
        retrieved_entries = system.hybrid_retriever.retrieve(query)
        retrieved = [entry_to_dict(e) for e in retrieved_entries]

        raw = system.answer_generator.generate_answer(query, retrieved_entries)
        if isinstance(raw, dict):
            answer    = raw.get("answer", "")
            reasoning = raw.get("reasoning", "")
        else:
            answer    = str(raw)
            reasoning = ""

        log = {
            "db_path":            tmp_dir,
            "condition":          condition,
            "conv_entries":       conv_entries,
            "stored_memories":    all_stored,
            "retrieved_memories": retrieved,
            "answer":             answer,
            "reasoning":          reasoning,
        }
        return retrieved, all_stored, answer, reasoning, log

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# log saving
# ---------------------------------------------------------------------------

def save_log(entry: dict, biased_log: dict, neutral_log: dict) -> Path:
    bias_dim     = entry.get("bias_dimension", "unknown")
    domain       = entry.get("domain",         "unknown")
    scenario_num = entry.get("scenario_num",   "unknown")
    attempt_num  = entry.get("attempt_num",    "unknown")

    log_dir = LOG_DIR / bias_dim
    log_dir.mkdir(parents=True, exist_ok=True)

    path = log_dir / f"{domain}_{scenario_num}_{attempt_num}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "bias_dimension": bias_dim,
            "domain":         domain,
            "scenario_num":   scenario_num,
            "attempt_num":    attempt_num,
            "biased":         biased_log,
            "neutral":        neutral_log,
        }, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# per-file processing
# ---------------------------------------------------------------------------

def process_file(
    input_path: Path,
    neutral_path: Path | None,
    api_key: str | None,
    resume: bool,
) -> None:
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
        domain       = entry.get("domain",       "?")
        scenario_num = entry.get("scenario_num", "?")
        attempt_num  = entry.get("attempt_num",  "?")
        label = f"{domain}/{scenario_num}/{attempt_num}"
        key   = (domain, scenario_num, attempt_num)

        print(f"  [{idx+1:3d}/{len(scenarios)}] {label}", end=" ", flush=True)

        prior           = prior_by_key.get(key, {})
        neutral_entries = neutral_by_key.get(key, {}).get("neutral_conversation", []) or []

        # ── biased (reuse only if prior has non-empty retrieval) ─────────────
        prior_biased = prior.get("biased_retrieved_memory") or []
        if prior_biased:
            biased_retrieved = prior_biased
            biased_all       = prior.get("biased_all_stored_memory") or []
            biased_answer    = prior.get("biased_answer", "") or ""
            biased_log       = None
            biased_tag       = "B:reuse"
        else:
            biased_retrieved, biased_all, biased_answer, _br, biased_log = run_condition(
                entry, "biased",
                entry.get("biased_conversation", []) or [],
                api_key,
            )
            biased_tag = "B:run"

        # ── neutral (reuse only if prior has non-empty retrieval) ────────────
        prior_neutral = prior.get("neutral_retrieved_memory") or []
        if prior_neutral:
            neutral_retrieved = prior_neutral
            neutral_all       = prior.get("neutral_all_stored_memory") or []
            neutral_answer    = prior.get("neutral_answer", "") or ""
            neutral_log       = None
            neutral_tag       = "N:reuse"
        else:
            neutral_retrieved, neutral_all, neutral_answer, _nr, neutral_log = run_condition(
                entry, "neutral", neutral_entries, api_key,
            )
            neutral_tag = "N:run"

        output_entry = copy.deepcopy(entry)
        output_entry["neutral_conversation"]      = neutral_entries
        output_entry["biased_all_stored_memory"]  = biased_all
        output_entry["biased_retrieved_memory"]   = biased_retrieved
        output_entry["biased_answer"]             = biased_answer
        output_entry["neutral_all_stored_memory"] = neutral_all
        output_entry["neutral_retrieved_memory"]  = neutral_retrieved
        output_entry["neutral_answer"]            = neutral_answer
        enriched.append(output_entry)

        if biased_log is not None or neutral_log is not None:
            log_path = save_log(entry, biased_log or {"reused": True}, neutral_log or {"reused": True})
            log_str  = f"  log:{log_path.name}"
        else:
            log_str = "  (both reused)"

        print(
            f"[{biased_tag} {neutral_tag}]"
            f" stored:{len(biased_all)}/{len(neutral_all)}"
            f"  retrieved:{len(biased_retrieved)}/{len(neutral_retrieved)}"
            f"{log_str}"
        )

        # incremental write so partial progress survives crashes
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved {len(enriched)} scenarios → {out_path}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SimpleMem retrieved memories for a bias dimension."
    )
    parser.add_argument("bias_dimension", help="e.g. speed_impatience")
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI-compatible API key (defaults to OPENAI_API_KEY env var)",
    )
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
    print(f"LLM / flags  : SimpleMem config defaults (see Memory/SimpleMem/config.py)")
    print(f"Resume       : {args.resume}")
    print(f"Isolation    : fresh temp LanceDB dir per scenario × condition (auto-deleted)")

    process_file(input_path, neutral_path if neutral_path.exists() else None, args.api_key, args.resume)

    print("\nAll done.")


if __name__ == "__main__":
    main()

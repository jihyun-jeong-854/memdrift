#!/usr/bin/env python3
"""Ingest the generated conversations into MemPalace and record what it retrieves.

MemPalace mines a plain-text transcript into memory entries with regex-based
chunking and retrieves them with BM25 keyword search. Both stages are
API-free — there is no extraction LLM and no embedding model, so no model
selection is possible or needed here.

Adds four fields per benchmark entry:

  - biased_all_stored_memory  / biased_retrieved_memory
  - neutral_all_stored_memory / neutral_retrieved_memory

Isolation: each entry x condition gets its own temporary palace_path (local
SQLite/Chroma), deleted afterwards, so nothing bleeds across entries.

All original columns are preserved unchanged.

Usage:
  python -m memory_construction.build_mempalace speed_impatience
  python -m memory_construction.build_mempalace speed_impatience --extract-mode exchange

Outputs:
  data/memories/mempalace/<bias_dimension>.json                            — enriched entries
  results/memory_logs/mempalace/<bias>/<domain>_<scenario>_<attempt>.json  — stored + retrieved trace
"""

import argparse
import copy
import json
import shutil
import sys
import tempfile
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

try:
    from mempalace.convo_miner import mine_convos
    from mempalace.palace import get_collection
    from mempalace.searcher import search_memories
except ImportError:
    print("ERROR: mempalace not installed. Activate the mempalace conda env.", file=sys.stderr)
    sys.exit(1)

WING          = "bench"
EXTRACT_MODE  = "general"   # "general" | "exchange"

INPUT_DIR  = CONVERSATION_DIR
OUTPUT_DIR = MEMORY_DIR / "mempalace"
LOG_DIR    = MEMORY_LOG_DIR / "mempalace"


# ---------------------------------------------------------------------------
# dialogue → MemPalace transcript  (ported from run_mempalace_final.py)
# ---------------------------------------------------------------------------

def dialogue_to_text(messages: list) -> str:
    lines = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role    = (msg.get("role") or "").strip().lower()
        content = msg.get("content", "") or ""
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        content = str(content).strip()
        if not content:
            continue
        if role == "user":
            for i, line in enumerate(content.splitlines()):
                lines.append(("> " if i == 0 else "  ") + line)
        elif role == "assistant":
            lines.extend(content.splitlines())
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# add + search for one condition
# ---------------------------------------------------------------------------

def run_condition(
    entry: dict,
    condition: str,
    conv_entries: list,
    extract_mode: str,
) -> tuple[list, list, dict]:
    """
    For each entry in conv_entries (a list of {<condition>_memory_idx, conversation}
    dicts) write one transcript file, then mine_convos → search_memories.
    Uses a fresh temp palace_path; deleted on exit.
    Returns (retrieved: list, all_stored: list, log: dict).
    """
    if not conv_entries:
        return [], [], {
            "palace_path": None,
            "wing": WING,
            "conv_entries": [],
            "mine_result": {},
            "all_stored_memories": [],
            "retrieved_memories": [],
        }

    tmp_palace = tempfile.mkdtemp(prefix=f"mempalace_bench_{condition}_")
    tmp_source = Path(tmp_palace) / "source"
    tmp_source.mkdir()

    try:
        # --- write one transcript file per memory entry ---
        idx_key = f"{condition}_memory_idx"
        written_files: list = []
        total_chars = 0
        for i, ce in enumerate(conv_entries):
            messages = (ce or {}).get("conversation") or []
            transcript_text = dialogue_to_text(messages)
            if not transcript_text.strip():
                continue
            mem_idx = (ce or {}).get(idx_key, i)
            fname = f"{condition}_memory_{mem_idx}.txt"
            fpath = tmp_source / fname
            fpath.write_text(transcript_text, encoding="utf-8")
            written_files.append(str(fpath))
            total_chars += len(transcript_text)

        if not written_files:
            return [], [], {
                "palace_path": tmp_palace,
                "wing": WING,
                "conv_entries": conv_entries,
                "mine_result": {},
                "all_stored_memories": [],
                "retrieved_memories": [],
            }

        # --- mine into palace ---
        mine_convos(
            convo_dir=str(tmp_source),
            palace_path=tmp_palace,
            wing=WING,
            agent="generate_mempalace_data",
            limit=0,
            dry_run=False,
            extract_mode=extract_mode,
        )

        # --- all stored memories ---
        all_stored: list = []
        try:
            col = get_collection(tmp_palace, create=False)
            raw_all = col.get(include=["documents", "metadatas"])
            for doc, meta in zip(raw_all.get("documents") or [], raw_all.get("metadatas") or []):
                text = (doc or "").strip()
                if not text:
                    continue
                all_stored.append({
                    "memory":       text,
                    "memory_type":  (meta or {}).get("room", "unknown"),
                    "wing":         (meta or {}).get("wing"),
                    "room":         (meta or {}).get("room"),
                    "source_file":  (meta or {}).get("source_file"),
                })
        except Exception:
            pass

        # --- search ---
        query   = entry.get("user_prompt", "")
        raw     = search_memories(
            query=query,
            palace_path=tmp_palace,
            wing=WING,
            room=None,
            max_distance=0.0,
        )
        results = raw.get("results", []) if isinstance(raw, dict) else []
        retrieved = _format_results(results)

        log = {
            "palace_path":       tmp_palace,
            "wing":              WING,
            "extract_mode":      extract_mode,
            "transcript_chars":  total_chars,
            "conv_entries":      conv_entries,
            "mine_result": {
                "source_files": written_files,
            },
            "all_stored_memories": all_stored,
            "retrieved_memories":  retrieved,
        }
        return retrieved, all_stored, log

    finally:
        shutil.rmtree(tmp_palace, ignore_errors=True)


def _format_results(results: list) -> list:
    formatted = []
    for idx, r in enumerate(results):
        text = (r.get("text") or "").strip()
        if not text:
            continue
        formatted.append({
            "id":           f"hit_{idx + 1}",
            "memory":       text,
            "score":        r.get("similarity"),
            "score_type":   "similarity",
            "memory_type":  r.get("room", "unknown"),
            "wing":         r.get("wing"),
            "room":         r.get("room"),
            "matched_via":  r.get("matched_via"),
            "source_file":  r.get("source_file"),
        })
    return formatted


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

def process_file(input_path: Path, neutral_path: Path | None, extract_mode: str, resume: bool):
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
        biased_entries  = entry.get("biased_conversation", []) or []
        neutral_entries = neutral_by_key.get(key, {}).get("neutral_conversation", []) or []

        # ── biased (reuse only if prior has non-empty retrieval) ─────────────
        prior_biased = prior.get("biased_retrieved_memory") or []
        if prior_biased:
            biased_retrieved = prior_biased
            biased_all       = prior.get("biased_all_stored_memory") or []
            biased_log       = None
            biased_tag       = "B:reuse"
        else:
            biased_retrieved, biased_all, biased_log = run_condition(
                entry, "biased", biased_entries, extract_mode,
            )
            biased_tag = "B:run"

        # ── neutral (reuse only if prior has non-empty retrieval) ────────────
        prior_neutral = prior.get("neutral_retrieved_memory") or []
        if prior_neutral:
            neutral_retrieved = prior_neutral
            neutral_all       = prior.get("neutral_all_stored_memory") or []
            neutral_log       = None
            neutral_tag       = "N:reuse"
        else:
            neutral_retrieved, neutral_all, neutral_log = run_condition(
                entry, "neutral", neutral_entries, extract_mode,
            )
            neutral_tag = "N:run"

        output_entry = copy.deepcopy(entry)
        output_entry["neutral_conversation"]      = neutral_entries
        output_entry["biased_all_stored_memory"]  = biased_all
        output_entry["biased_retrieved_memory"]   = biased_retrieved
        output_entry["neutral_all_stored_memory"] = neutral_all
        output_entry["neutral_retrieved_memory"]  = neutral_retrieved
        enriched.append(output_entry)

        if biased_log is not None or neutral_log is not None:
            log_path = save_log(entry, biased_log or {"reused": True}, neutral_log or {"reused": True})
            log_str  = f"  log:{log_path.name}"
        else:
            log_str = "  (both reused)"

        print(f"[{biased_tag} {neutral_tag}] biased:{len(biased_retrieved)} neutral:{len(neutral_retrieved)}{log_str}")

        # incremental write so partial progress survives crashes
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved {len(enriched)} scenarios → {out_path}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate MemPalace retrieved memories for a bias dimension.")
    parser.add_argument("bias_dimension", help="e.g. speed_impatience")
    parser.add_argument(
        "--extract-mode", default=EXTRACT_MODE,
        choices=["general", "exchange"],
        help="MemPalace mining mode: 'general' (decisions/preferences/facts) or 'exchange' (Q+A pairs). Default: general",
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
    print(f"Extract mode : {args.extract_mode}")
    print(f"Search       : BM25, default limit")
    print(f"Resume       : {args.resume}")
    print(f"Isolation    : fresh temp palace per scenario × condition (auto-deleted)")

    process_file(input_path, neutral_path if neutral_path.exists() else None, args.extract_mode, args.resume)

    print("\nAll done.")


if __name__ == "__main__":
    main()

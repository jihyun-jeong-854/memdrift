#!/usr/bin/env python3
"""Fetch the MemDrift benchmark data from Hugging Face into data/.

The Hugging Face dataset is the single source of truth for the benchmark; this
repo holds only code, so that the two cannot drift apart. The download lays the
files out exactly where common/paths.py expects them:

    data/scenarios/<bias>.json                 105 verified entries (incl. top-2 annotations)
    data/rubrics/all_params/<bias>.json        objective rubric, every critical parameter
    data/conversations/<bias>.json             + biased_conversation
    data/conversations/<bias>_neutral.json     + neutral_conversation
    data/memories/<arch>/<bias>.json           + retrieved memory per architecture

Usage:
    python scripts/download_data.py                      # everything
    python scripts/download_data.py --subset scenarios   # direct-injection setting only
    python scripts/download_data.py --repo other-org/memdrift
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA_ROOT

DEFAULT_REPO = os.getenv("MEMDRIFT_HF_REPO", "MemDrift/MemDrift")

# Which directories each subset needs. The direct-injection setting needs only
# `scenarios`; the memory-architecture setting needs `memories` as well.
SUBSETS = {
    "scenarios": ["scenarios", "rubrics"],     # setting 1 + the objective rubric
    "conversations": ["conversations"],
    "memories": ["memories"],
    "all": ["scenarios", "rubrics", "conversations", "memories"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"Hugging Face dataset repo id (default: {DEFAULT_REPO})")
    parser.add_argument("--subset", default="all", choices=sorted(SUBSETS),
                        help="Which part of the data to fetch (default: all)")
    parser.add_argument("--revision", default=None,
                        help="Branch, tag or commit to pin (default: the main branch)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help=f"Destination directory (default: {DATA_ROOT})")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )

    patterns = [f"{d}/**" for d in SUBSETS[args.subset]]
    args.data_root.mkdir(parents=True, exist_ok=True)

    print(f"Repo     : {args.repo}")
    print(f"Subset   : {args.subset} ({', '.join(SUBSETS[args.subset])})")
    print(f"Target   : {args.data_root}")

    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=patterns,
        local_dir=str(args.data_root),
    )

    for d in SUBSETS[args.subset]:
        files = sorted((args.data_root / d).rglob("*.json"))
        print(f"  {d}: {len(files)} files")

    print("\nDone.")


if __name__ == "__main__":
    main()

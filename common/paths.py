"""Canonical filesystem layout for MemDrift.

Every stage reads and writes through this module, so no absolute path is baked
into the code. Override either root:

    MEMDRIFT_DATA     benchmark artifacts   (default: <repo>/data)
    MEMDRIFT_RESULTS  run outputs           (default: <repo>/results)

    data/scenarios/<bias>.json             105 entries, 21 per bias dimension
    data/rubrics/all_params/<bias>.json    degradation-rate rubric
    data/conversations/<bias>.json         + biased_conversation
    data/conversations/<bias>_neutral.json + neutral_conversation
    data/memories/<mem_arch>/<bias>.json   + {biased,neutral}_retrieved_memory

    results/inference/<setting>/<bias>_<model>_<reasoning>.json
    results/scores/objective/<variant>/
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


DATA_ROOT = Path(os.getenv("MEMDRIFT_DATA", REPO_ROOT / "data")).expanduser()
SCENARIO_DIR = DATA_ROOT / "scenarios"
CONVERSATION_DIR = DATA_ROOT / "conversations"
MEMORY_DIR = DATA_ROOT / "memories"
RUBRIC_DIR = DATA_ROOT / "rubrics"

RESULTS_ROOT = Path(os.getenv("MEMDRIFT_RESULTS", REPO_ROOT / "results")).expanduser()
INFERENCE_DIR = RESULTS_ROOT / "inference"
SCORE_DIR = RESULTS_ROOT / "scores"

# The three memory frameworks evaluated in the paper.
MEMORY_ARCHITECTURES = ("mem0", "mempalace", "simplemem")

# The five bias dimensions (configs/bias_dimensions.json).
BIAS_DIMENSIONS = (
    "speed_impatience",
    "resource_frugality",
    "minimalism_conciseness",
    "risk_permissiveness",
    "autonomy_self_reliance",
)


ENV_FILE = REPO_ROOT / ".env"


def scenario_file(bias_dimension):
    """Verified benchmark entries for one bias dimension."""
    return SCENARIO_DIR / f"{bias_dimension}.json"


def conversation_file(bias_dimension, condition="biased"):
    """Conversation-expanded entries. condition is 'biased' or 'neutral'."""
    if condition == "biased":
        return CONVERSATION_DIR / f"{bias_dimension}.json"
    if condition == "neutral":
        return CONVERSATION_DIR / f"{bias_dimension}_neutral.json"
    raise ValueError(f"condition must be 'biased' or 'neutral', got {condition!r}")


def rubric_file(bias_dimension, variant="all_params"):
    """Objective-metric rubric: parameter severity orderings for one bias dimension."""
    return RUBRIC_DIR / variant / f"{bias_dimension}.json"


def memory_file(bias_dimension, mem_arch):
    """Entries enriched with what `mem_arch` stored and retrieved."""
    return MEMORY_DIR / mem_arch / f"{bias_dimension}.json"


def result_name(bias_dimension, model, reasoning):
    """Filename shared by inference outputs and judged outputs."""
    safe_model = str(model).split("/")[-1]
    return f"{bias_dimension}_{safe_model}_{reasoning}.json"


def inference_file(bias_dimension, model, reasoning, setting="direct"):
    """`setting` is 'direct' (direct memory injection) or a memory arch name."""
    return INFERENCE_DIR / setting / result_name(bias_dimension, model, reasoning)


def score_file(bias_dimension, model, reasoning, setting="direct"):
    return SCORE_DIR / setting / result_name(bias_dimension, model, reasoning)

#!/usr/bin/env bash
# Setting 2 — memory frameworks, end to end: inference -> objective score.
#
# Loops the inference over memory frameworks for one (bias dimension, model)
# pair, then scores them together. The system prompt is filled with what each
# architecture actually retrieved for the professional query, so
# data/memories/<arch>/<bias>.json must already exist
# (download it from the Hugging Face dataset: scripts/download_data.py).
#
# Usage:
#   bash scripts/run_memory_frameworks.sh                        # bias = speed_impatience
#   bash scripts/run_memory_frameworks.sh resource_frugality     # override bias
#   PROVIDER=openrouter MODEL=google/gemini-3.1-pro-preview \
#       bash scripts/run_memory_frameworks.sh speed_impatience
#   MEM_ARCHS="mem0" bash scripts/run_memory_frameworks.sh       # one architecture
#   RESUME=1 bash scripts/run_memory_frameworks.sh               # reuse non-'na' cells

set -u

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-5.4}"
REASONING="${REASONING:-medium}"
N_RUNS="${N_RUNS:-5}"
BIAS="${1:-speed_impatience}"
MEM_ARCHS="${MEM_ARCHS:-mem0 mempalace simplemem}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${MEMDRIFT_RESULTS:-${REPO_ROOT}/results}"

SAFE_MODEL="${MODEL##*/}"
RESULT_NAME="${BIAS}_${SAFE_MODEL}_${REASONING}.json"

RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG="--resume"
fi

cd "$REPO_ROOT"

scored=""
for mem_arch in $MEM_ARCHS; do
    printf "\n================================================================\n"
    printf " %s | bias = %s | model = %s_%s\n" "$mem_arch" "$BIAS" "$SAFE_MODEL" "$REASONING"
    printf "================================================================\n"

    python -m drift_inference.inference_memory \
        --provider "$PROVIDER" \
        --bias "$BIAS" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        --mem_arch "$mem_arch" \
        --n_runs "$N_RUNS" \
        $RESUME_FLAG || { echo "[skip $mem_arch] inference failed" >&2; continue; }

    inference_json="${RESULTS_ROOT}/inference/${mem_arch}/${RESULT_NAME}"
    if [[ ! -f "$inference_json" ]]; then
        echo "[skip $mem_arch] inference output missing: $inference_json" >&2
        continue
    fi
    scored="${scored} ${mem_arch}"
done

if [[ -z "${scored// }" ]]; then
    echo "No architectures produced inference output." >&2
    exit 1
fi

# One scoring pass over the architectures that ran. objective_drift prints the
# comparison table itself and writes per_scenario / by_bias / by_setting /
# by_model under results/scores/objective/<variant>/.
printf "\n================================================================\n"
printf " scoring: %s | bias = %s\n" "${scored# }" "$BIAS"
printf "================================================================\n"
python -m drift_scoring.objective_drift \
    --settings $scored \
    --models "$MODEL" \
    --reasoning "$REASONING" \
    --bias-dimensions "$BIAS"

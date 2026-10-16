#!/usr/bin/env bash
# Setting 1 — direct memory injection, end to end: inference -> objective score.
#
# Loops the inference over bias dimensions for one model, then scores every
# dimension in a single pass. Memory sets go straight into the system prompt;
# no memory system is involved.
#
# Usage:
#   bash scripts/run_direct_injection.sh                            # defaults below
#   PROVIDER=openai MODEL=gpt-5.4 bash scripts/run_direct_injection.sh
#   BIASES="speed_impatience risk_permissiveness" bash scripts/run_direct_injection.sh
#   RESUME=1 bash scripts/run_direct_injection.sh                   # reuse non-'na' cells

set -u

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-5.4}"
REASONING="${REASONING:-medium}"
N_RUNS="${N_RUNS:-5}"
BIASES="${BIASES:-speed_impatience resource_frugality minimalism_conciseness risk_permissiveness autonomy_self_reliance}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${MEMDRIFT_RESULTS:-${REPO_ROOT}/results}"
SETTING="direct"

SAFE_MODEL="${MODEL##*/}"

RESUME_FLAG=""
if [[ "${RESUME:-0}" == "1" ]]; then
    RESUME_FLAG="--resume"
fi

cd "$REPO_ROOT"

scored=""
for bias in $BIASES; do
    printf "\n================================================================\n"
    printf " %s | bias = %s\n" "$SETTING" "$bias"
    printf "================================================================\n"

    python -m drift_inference.inference_direct \
        --provider "$PROVIDER" \
        --bias "$bias" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        --n_runs "$N_RUNS" \
        $RESUME_FLAG || { echo "[skip $bias] inference failed" >&2; continue; }

    inference_json="${RESULTS_ROOT}/inference/${SETTING}/${bias}_${SAFE_MODEL}_${REASONING}.json"
    if [[ ! -f "$inference_json" ]]; then
        echo "[skip $bias] inference output missing: $inference_json" >&2
        continue
    fi
    scored="${scored} ${bias}"
done

if [[ -z "${scored// }" ]]; then
    echo "No bias dimensions produced inference output." >&2
    exit 1
fi

# One scoring pass over everything that ran. objective_drift prints the
# per-dimension table itself and writes per_scenario / by_bias / by_setting /
# by_model under results/scores/objective/<variant>/.
printf "\n================================================================\n"
printf " scoring: %s\n" "${scored# }"
printf "================================================================\n"
python -m drift_scoring.objective_drift \
    --settings "$SETTING" \
    --models "$MODEL" \
    --reasoning "$REASONING" \
    --bias-dimensions $scored

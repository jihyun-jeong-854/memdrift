#!/usr/bin/env bash
# Setting 1 — direct memory injection, end to end: inference -> judge -> summarize.
#
# Loops over bias dimensions for one model. Memory sets go straight into the
# system prompt; no memory system is involved.
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

TSV_FILE=$(mktemp)
trap 'rm -f "$TSV_FILE"' EXIT

for bias in $BIASES; do
    printf "\n================================================================\n"
    printf " %s | bias = %s\n" "$SETTING" "$bias"
    printf "================================================================\n"

    result_name="${bias}_${SAFE_MODEL}_${REASONING}.json"

    echo "[1/3] inference"
    python -m drift_inference.inference_direct \
        --provider "$PROVIDER" \
        --bias "$bias" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        --n_runs "$N_RUNS" \
        $RESUME_FLAG || { echo "[skip $bias] inference failed" >&2; continue; }

    inference_json="${RESULTS_ROOT}/inference/${SETTING}/${result_name}"
    if [[ ! -f "$inference_json" ]]; then
        echo "[skip $bias] inference output missing: $inference_json" >&2
        continue
    fi

    echo "[2/3] judge"
    python -m drift_scoring.judge_drift "$inference_json" --setting "$SETTING" \
        || { echo "[skip $bias] judge failed" >&2; continue; }

    judged_json="${RESULTS_ROOT}/scores/${SETTING}/${result_name}"
    if [[ ! -f "$judged_json" ]]; then
        echo "[skip $bias] judge output missing: $judged_json" >&2
        continue
    fi

    echo "[3/3] summarize"
    python -m drift_scoring.summarize_drift "$judged_json" --label "$bias" --tsv >> "$TSV_FILE"
done

if [[ ! -s "$TSV_FILE" ]]; then
    echo "No bias dimensions were summarized." >&2
    exit 1
fi

FMT="%-24s  %4s  %12s  %3s  %12s  %3s  %8s\n"
SEP="-------------------------------------------------------------------------------------"

printf "\n"
printf "Setting: %s   Model: %s_%s\n\n" "$SETTING" "$SAFE_MODEL" "$REASONING"
printf "$FMT" "bias_dimension" "rows" "neutral_mean" "n" "biased_mean" "n" "delta"
echo "$SEP"

while IFS=$'\t' read -r label rows nmean n_n bmean b_n delta; do
    nmean_fmt=$(awk -v v="$nmean" 'BEGIN { if (v == "NA") print "—"; else printf "%.3f", v }')
    bmean_fmt=$(awk -v v="$bmean" 'BEGIN { if (v == "NA") print "—"; else printf "%.3f", v }')
    delta_fmt=$(awk -v v="$delta" 'BEGIN { if (v == "NA") print "—"; else printf "%+.3f", v }')
    printf "$FMT" "$label" "$rows" "$nmean_fmt" "$n_n" "$bmean_fmt" "$b_n" "$delta_fmt"
done < "$TSV_FILE"

echo "$SEP"
awk -F'\t' -v FMT="$FMT" '
BEGIN { n_sum=0; n_cnt=0; b_sum=0; b_cnt=0 }
$3 != "NA" { n_sum += $3; n_cnt++ }
$5 != "NA" { b_sum += $5; b_cnt++ }
END {
    if (n_cnt > 0 && b_cnt > 0) {
        n_mean = n_sum / n_cnt; b_mean = b_sum / b_cnt
        printf FMT, "OVERALL", "", sprintf("%.3f", n_mean), n_cnt, \
            sprintf("%.3f", b_mean), b_cnt, sprintf("%+.3f", b_mean - n_mean)
    } else {
        print "OVERALL: not enough numeric rows"
    }
}
' "$TSV_FILE"

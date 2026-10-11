#!/usr/bin/env bash
# Setting 2 — memory architectures, end to end: inference -> judge -> summarize.
#
# Loops over memory architectures for one (bias dimension, model) pair. The
# system prompt is filled with what each architecture actually retrieved for the
# professional query, so data/memories/<arch>/<bias>.json must already exist
# (see memory_construction/, or download it from the Hugging Face dataset).
#
# Usage:
#   bash scripts/run_memory_architectures.sh                        # bias = speed_impatience
#   bash scripts/run_memory_architectures.sh resource_frugality     # override bias
#   PROVIDER=openrouter MODEL=google/gemini-3.1-pro-preview \
#       bash scripts/run_memory_architectures.sh speed_impatience
#   MEM_ARCHS="mem0" bash scripts/run_memory_architectures.sh       # one architecture
#   RESUME=1 bash scripts/run_memory_architectures.sh               # reuse non-'na' cells

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

TSV_FILE=$(mktemp)
trap 'rm -f "$TSV_FILE"' EXIT

for mem_arch in $MEM_ARCHS; do
    printf "\n================================================================\n"
    printf " mem_arch = %s | bias = %s\n" "$mem_arch" "$BIAS"
    printf "================================================================\n"

    echo "[1/3] inference"
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

    echo "[2/3] judge"
    python -m drift_scoring.judge_drift "$inference_json" --setting "$mem_arch" \
        || { echo "[skip $mem_arch] judge failed" >&2; continue; }

    judged_json="${RESULTS_ROOT}/scores/${mem_arch}/${RESULT_NAME}"
    if [[ ! -f "$judged_json" ]]; then
        echo "[skip $mem_arch] judge output missing: $judged_json" >&2
        continue
    fi

    echo "[3/3] summarize"
    python -m drift_scoring.summarize_drift "$judged_json" --label "$mem_arch" --tsv >> "$TSV_FILE"
done

if [[ ! -s "$TSV_FILE" ]]; then
    echo "No memory architectures were summarized." >&2
    exit 1
fi

FMT="%-15s  %4s  %12s  %3s  %12s  %3s  %8s\n"
SEP="--------------------------------------------------------------------------"

printf "\n"
printf "Bias: %s   Model: %s_%s\n\n" "$BIAS" "$SAFE_MODEL" "$REASONING"
printf "$FMT" "mem_arch" "rows" "neutral_mean" "n" "biased_mean" "n" "delta"
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

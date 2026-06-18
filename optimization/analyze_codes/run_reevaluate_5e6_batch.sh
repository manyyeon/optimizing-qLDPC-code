#!/usr/bin/env bash
# Run all ten 5e6 re-evaluations for one method/C pair.
#
# Usage:
#   bash run_reevaluate_5e6_batch.sh beam 0 4
#   bash run_reevaluate_5e6_batch.sh greedy 3 4

set -uo pipefail

METHOD="${1:?Usage: $0 <beam|greedy> <C:0-3> [workers]}"
C="${2:?Usage: $0 <beam|greedy> <C:0-3> [workers]}"
WORKERS="${3:-4}"

RESULTS_ROOT="optimization/results/logical_guided_absolute_score_gamma01_slack1_repeated_runs"
LOG_ROOT="run_logs/logical_guided_absolute_score_gamma01_slack1_reeval_5e6/${METHOD}/C${C}"
if [[ -f "optimization/analyze_codes/reevaluate_best_5e6.py" ]]; then
    PYTHON_SCRIPT="optimization/analyze_codes/reevaluate_best_5e6.py"
elif [[ -f "reevaluate_best_5e6.py" ]]; then
    PYTHON_SCRIPT="reevaluate_best_5e6.py"
else
    echo "ERROR: reevaluate_best_5e6.py was not found." >&2
    exit 2
fi

mkdir -p "${LOG_ROOT}"

overall_status=0

for i in $(seq 1 10); do
    stamp="$(date +%Y%m%d_%H%M%S)"
    log_file="${LOG_ROOT}/${METHOD}_C${C}_run${i}_5e6_${stamp}.log"

    echo
    echo "================================================================"
    echo "Starting ${METHOD}, C=${C}, run=${i}"
    echo "Log: ${log_file}"
    echo "================================================================"

    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH="$(pwd)" \
    python3 -u "${PYTHON_SCRIPT}" \
        --results-root "${RESULTS_ROOT}" \
        --method "${METHOD}" \
        -C "${C}" \
        --run "${i}" \
        --budget 5000000 \
        --workers "${WORKERS}" \
        --batch-size 50000 \
        --selection distance_then_ler \
        --update-best-selection \
        2>&1 | tee "${log_file}"

    status=${PIPESTATUS[0]}
    if [[ ${status} -ne 0 ]]; then
        echo "FAILED: ${METHOD}, C=${C}, run=${i}, exit=${status}" \
            | tee -a "${log_file}"
        overall_status=1
    else
        echo "COMPLETED: ${METHOD}, C=${C}, run=${i}" \
            | tee -a "${log_file}"
    fi
done

exit "${overall_status}"

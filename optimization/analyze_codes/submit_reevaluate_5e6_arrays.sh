#!/usr/bin/env bash
set -euo pipefail

mkdir -p run_logs/slurm
SBATCH_FILE="optimization/analyze_codes/reevaluate_5e6_array.sbatch"

for method in beam greedy; do
  for C in 0 1 2 3; do
    sbatch \
      --job-name="reeval_${method}_C${C}" \
      --export=ALL,METHOD="${method}",C="${C}" \
      "${SBATCH_FILE}"
  done
done

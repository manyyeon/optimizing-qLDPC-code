#!/usr/bin/env bash
#SBATCH --account=2016435
#SBATCH --partition=general
#SBATCH --qos=easley
#SBATCH --array=1-10%1
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=run_logs/slurm/%x_%A_%a.out
#SBATCH --error=run_logs/slurm/%x_%A_%a.err

set -euo pipefail

: "${METHOD:?Submit with --export=ALL,METHOD=beam|greedy,C=0|1|2|3}"
: "${C:?Submit with --export=ALL,METHOD=beam|greedy,C=0|1|2|3}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p run_logs/slurm
source venv312/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}"

RUN="${SLURM_ARRAY_TASK_ID}"
WORKERS="${SLURM_CPUS_PER_TASK}"

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Method: ${METHOD}"
echo "C: ${C}"
echo "Run: ${RUN}"
echo "Workers: ${WORKERS}"

python3 -u optimization/analyze_codes/reevaluate_best_5e6.py \
  --method "${METHOD}" \
  -C "${C}" \
  --run "${RUN}" \
  --budget 5000000 \
  --workers "${WORKERS}" \
  --batch-size 10000 \
  --selection distance_then_ler \
  --update-best-selection

echo "Finished: $(date)"

#!/usr/bin/env bash
#SBATCH --job-name=agop-analyze
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/agop_analyze_%j.out
#SBATCH --error=results/slurm/agop_analyze_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/slurm

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "Started at $(date)"
for dir in results/main results/ablations/* results/sensitivity/*; do
  if [[ -d "$dir" ]]; then
    echo "Analyzing $dir"
    python analyze_results.py --results_dir "$dir"
  fi
done
python scripts/summarize_paper_results.py --results_root results --out results/paper_summary.md
echo "Finished at $(date)"

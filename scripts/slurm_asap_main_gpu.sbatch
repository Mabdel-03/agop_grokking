#!/usr/bin/env bash
#SBATCH --job-name=agop-asap-main
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/agop_asap_main_%j.out
#SBATCH --error=results/slurm/agop_asap_main_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/slurm
CONFIG="${CONFIG:-results/asap_best.yaml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing ASAP main config: $CONFIG" >&2
  exit 1
fi
echo "ASAP main job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "Started at $(date)"
python run_experiment.py --config "$CONFIG" --seeds 0 1 2 --results_dir results/asap_main --device cuda --no_progress
python analyze_results.py --results_dir results/asap_main
echo "Finished at $(date)"

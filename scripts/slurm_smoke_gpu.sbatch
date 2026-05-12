#!/usr/bin/env bash
#SBATCH --job-name=agop-smoke
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/agop_smoke_%j.out
#SBATCH --error=results/slurm/agop_smoke_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/slurm

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "Started at $(date)"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_name", torch.cuda.get_device_name(0))
PY

python run_experiment.py --config configs/smoke.yaml --device cuda --no_progress
python analyze_results.py --results_dir results/smoke
echo "Finished at $(date)"

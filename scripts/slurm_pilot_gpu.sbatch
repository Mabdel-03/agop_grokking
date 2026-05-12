#!/usr/bin/env bash
#SBATCH --job-name=agop-pilot
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/agop_pilot_%j.out
#SBATCH --error=results/slurm/agop_pilot_%j.err

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

python -m py_compile run_experiment.py analyze_results.py src/*.py scripts/*.py
python run_experiment.py --config configs/pilot.yaml --device cuda --no_progress
python scripts/generate_manifests.py --best_config results/pilot/best_config.yaml
echo "Finished at $(date)"

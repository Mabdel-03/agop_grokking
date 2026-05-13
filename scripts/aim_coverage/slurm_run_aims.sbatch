#!/usr/bin/env bash
#SBATCH --job-name=agop-aim-coverage
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/agop_aim_coverage_%j.out
#SBATCH --error=results/slurm/agop_aim_coverage_%j.err

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

python -m py_compile run_experiment.py analyze_results.py src/*.py

echo "=== Aim 1: block-NTK ==="
python run_experiment.py --config configs/aim1_block_ntk.yaml --device cuda --no_progress
python analyze_results.py --results_dir results/aim1_block_ntk

echo "=== Aim 2: held-out VCR ==="
python run_experiment.py --config configs/aim2_vcr.yaml --device cuda --no_progress
python analyze_results.py --results_dir results/aim2_vcr

echo "=== Aim 3: modular multiplication ==="
python run_experiment.py --config configs/aim3_mod_mult.yaml --device cuda --no_progress
python analyze_results.py --results_dir results/aim3_mod_mult

echo "=== Aim 3: matched RFM ==="
python scripts/aim_coverage/run_rfm.py --config configs/aim3_rfm.yaml
python analyze_results.py --results_dir results/aim3_rfm || echo "RFM analyze step optional"

echo "Finished at $(date)"

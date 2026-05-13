#!/usr/bin/env bash
# Preliminary runs that produced Table 1 and Figure 1 of the paper.
# Three seeds of the proven recipe (p=31, width 512, 50k steps).
#SBATCH --job-name=agop-preliminary
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm/preliminary_%j.out
#SBATCH --error=results/slurm/preliminary_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/slurm

echo "Preliminary-runs job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "Started at $(date)"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_name", torch.cuda.get_device_name(0))
PY

python -m py_compile run_experiment.py analyze_results.py src/*.py

python run_experiment.py --config configs/preliminary.yaml --device cuda --no_progress
python analyze_results.py --results_dir results/preliminary

echo "Finished at $(date)"

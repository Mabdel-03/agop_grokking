#!/usr/bin/env bash
#SBATCH --job-name=agop-asap
#SBATCH --partition=ou_bcs_low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/agop_asap_%A_%a.out
#SBATCH --error=results/slurm/agop_asap_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/slurm

MANIFEST="${MANIFEST:-results/asap_manifests/manifest.csv}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
echo "ASAP job ${SLURM_JOB_ID:-manual}, task ${TASK_ID}, on $(hostname)"
echo "Started at $(date)"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_name", torch.cuda.get_device_name(0))
PY
python scripts/run_asap_entry.py --manifest "$MANIFEST" --index "$TASK_ID" --device cuda
echo "Finished at $(date)"

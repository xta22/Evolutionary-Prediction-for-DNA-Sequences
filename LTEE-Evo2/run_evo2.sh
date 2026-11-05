#!/bin/bash
#SBATCH --job-name=evo2
#SBATCH --account=r01340
#SBATCH --partition=h100-single
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Load software environment
module purge
module load gnu/12.2.0
module load python/gpu/3.11.5
module load cudatoolkit/12.8

source "$HOME/envs/evo2/bin/activate"

# CUDA and model cache
export CUDA_HOME=/N/soft/rhel8/cuda/12.8
export HF_HOME=/N/scratch/$USER/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=/N/scratch/$USER/evo2_cache/torch

# Project directory
cd "$HOME/LTEE-Evo2"

echo "========================================"
echo "EVO2 GPU JOB"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
echo

nvidia-smi

# Run requested Python script
if [ "$#" -eq 0 ]; then
    echo "ERROR: No Python script provided."
    exit 1
fi

echo
echo "Running: $*"
echo

python -u "$@"

echo
echo "========================================"
echo "JOB COMPLETED"
echo "Finished: $(date)"
echo "========================================"

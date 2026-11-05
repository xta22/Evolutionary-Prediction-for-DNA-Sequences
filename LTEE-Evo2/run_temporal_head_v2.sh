#!/bin/bash
#SBATCH -A r01340
#SBATCH -p general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=temporal_head
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load gnu/12.2.0
module load python/gpu/3.11.5

source "$HOME/envs/evo2/bin/activate"

cd "$HOME/LTEE-Evo2"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -u train_temporal_head_v2.py

#!/bin/bash
#SBATCH --job-name=evo2_repr_shift
#SBATCH --account=r01340
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/evo2_repr_shift_%j.out
#SBATCH --error=logs/evo2_repr_shift_%j.err
set -euo pipefail
module purge
module load gnu/12.2.0
module load python/gpu/3.11.5
source "$HOME/envs/evo2/bin/activate"
cd "$HOME/LTEE-Evo2"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -u investigate_representation_shift.py --permutations 300

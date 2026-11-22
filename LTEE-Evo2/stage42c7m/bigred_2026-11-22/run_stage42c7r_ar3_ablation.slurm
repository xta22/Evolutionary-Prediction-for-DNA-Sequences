#!/bin/bash
#SBATCH --job-name=stage42c7r
#SBATCH --account=r00998
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=/N/u/xta1/BigRed200/stage42c7m/logs/%x-%j.out
#SBATCH --error=/N/u/xta1/BigRed200/stage42c7m/logs/%x-%j.err

set -euo pipefail
cd /N/u/xta1/BigRed200/stage42c7m
module load python/gpu/3.11.5
module load cudatoolkit/12.8
source "$HOME/envs/evo2/bin/activate"

echo "HOST=$(hostname)"
python --version
python -c 'import numpy; print("NumPy", numpy.__version__)'

python -u stage42c7r_ar3_high_n_fold_ablation.py \
  --input-dir "runs/9110885/input_bundle" \
  --embeddings-dir "runs/9110885/evo2_embeddings" \
  --output "runs/9110885/ar3_high_n_fold_ablation_fix2"

echo "RESULTS=$PWD/runs/9110885/ar3_high_n_fold_ablation_fix2"

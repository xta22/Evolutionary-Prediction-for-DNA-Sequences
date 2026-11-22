#!/bin/bash
#SBATCH --job-name=stage42c7p
#SBATCH --account=r00998
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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

python -u stage42c7p_target_frequency_uncertainty.py \
  --predictions "runs/9110885/target_clone_weight_sensitivity/heldout_site_predictions.csv" \
  --output "runs/9110885/target_frequency_uncertainty" \
  --draws 5000 \
  --seed 734201

echo "RESULTS=$PWD/runs/9110885/target_frequency_uncertainty"

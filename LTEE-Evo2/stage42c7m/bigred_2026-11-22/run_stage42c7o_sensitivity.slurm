#!/bin/bash
#SBATCH --job-name=stage42c7o
#SBATCH --account=r00998
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

RUN_DIR="$PWD/runs/9110885"
python -u stage42c7o_target_clone_weight_sensitivity.py \
  --input-dir "$RUN_DIR/input_bundle" \
  --embeddings-dir "$RUN_DIR/evo2_embeddings" \
  --output "$RUN_DIR/target_clone_weight_sensitivity"

echo "RESULTS=$RUN_DIR/target_clone_weight_sensitivity"

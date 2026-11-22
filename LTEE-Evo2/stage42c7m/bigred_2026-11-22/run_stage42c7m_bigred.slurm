#!/bin/bash
#SBATCH --job-name=stage42c7m
#SBATCH --account=r00998
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/N/u/xta1/BigRed200/stage42c7m/logs/%x-%j.out
#SBATCH --error=/N/u/xta1/BigRed200/stage42c7m/logs/%x-%j.err

set -euo pipefail
cd /N/u/xta1/BigRed200/stage42c7m

module purge
module load python/gpu/3.11.5
module load cudatoolkit/12.8
source "$HOME/envs/evo2/bin/activate"

echo "HOST=$(hostname)"
nvidia-smi
python -c 'import torch; print("torch", torch.__version__, "CUDA", torch.cuda.is_available()); assert torch.cuda.is_available()'
python -c 'from evo2 import Evo2; print("Evo2 import OK")'

RUN_DIR="$PWD/runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/input_bundle"
unzip -o v3c_stage42c7m_evo2_windows_transfer.zip -d "$RUN_DIR/input_bundle"

python -u stage42c7m_evo2_embed.py \
  --input-dir "$RUN_DIR/input_bundle" \
  --output "$RUN_DIR/evo2_embeddings" \
  --model evo2_7b \
  --layer blocks.28.mlp.l3

python -u stage42c7m_evo2_lopo_eval.py \
  --input-dir "$RUN_DIR/input_bundle" \
  --embeddings-dir "$RUN_DIR/evo2_embeddings" \
  --output "$RUN_DIR/lopo_results"

echo "RESULTS=$RUN_DIR/lopo_results"

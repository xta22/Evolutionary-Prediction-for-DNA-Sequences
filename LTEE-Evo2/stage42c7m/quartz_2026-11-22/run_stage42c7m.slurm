#!/bin/bash
#SBATCH --job-name=stage42c7m
#SBATCH --account=r01340
#SBATCH --partition=h100-single
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/N/u/xta1/Quartz/LTEE-Evo2/stage42c7m/logs/%x-%j.out
#SBATCH --error=/N/u/xta1/Quartz/LTEE-Evo2/stage42c7m/logs/%x-%j.err

set -euo pipefail
cd /N/u/xta1/Quartz/LTEE-Evo2/stage42c7m
module purge
module load gnu/12.2.0
module load python/gpu/3.11.5
module load cudatoolkit/12.8
source "$HOME/envs/evo2/bin/activate"
export CUDA_HOME=/N/soft/rhel8/cuda/12.8

echo "HOST=$(hostname)"
nvidia-smi
python -c 'import torch; print("torch", torch.__version__, "CUDA", torch.cuda.is_available()); assert torch.cuda.is_available()'

RUN_DIR="$PWD/runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/input_bundle"
unzip -o stage42c7m_evo2_windows_transfer.zip -d "$RUN_DIR/input_bundle"

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

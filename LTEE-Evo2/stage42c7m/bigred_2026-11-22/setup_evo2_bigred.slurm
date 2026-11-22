#!/bin/bash
#SBATCH --job-name=evo2-setup
#SBATCH --account=r00998
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/N/u/xta1/BigRed200/stage42c7m/logs/evo2setup-%j.out
#SBATCH --error=/N/u/xta1/BigRed200/stage42c7m/logs/evo2setup-%j.err

set -euo pipefail
cd /N/u/xta1/BigRed200/stage42c7m

module purge
module load python/gpu/3.11.5
module load cudatoolkit/12.8

echo "HOST=$(hostname)"
module list
nvidia-smi
nvcc --version
gcc --version

mkdir -p "$HOME/envs"
if [ ! -x "$HOME/envs/evo2/bin/python" ]; then
    python -m venv "$HOME/envs/evo2"
fi
source "$HOME/envs/evo2/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

MAX_JOBS=4 python -m pip install flash-attn==2.8.0.post2 \
    --no-build-isolation

python -m pip install evo2==0.6.0 triton==3.3.1 vtx==1.1.0

python -m pip check
python -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
python -c 'from evo2 import Evo2; print("Evo2 import OK")'

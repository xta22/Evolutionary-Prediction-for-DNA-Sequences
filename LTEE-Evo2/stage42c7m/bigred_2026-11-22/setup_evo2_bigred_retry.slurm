#!/bin/bash
#SBATCH --job-name=evo2-retry
#SBATCH --account=r00998
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/N/u/xta1/BigRed200/stage42c7m/logs/evo2retry-%j.out
#SBATCH --error=/N/u/xta1/BigRed200/stage42c7m/logs/evo2retry-%j.err

set -euo pipefail
cd /N/u/xta1/BigRed200/stage42c7m

module purge
module load python/gpu/3.11.5
module load cudatoolkit/12.8

source "$HOME/envs/evo2/bin/activate"

WHEEL="flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/$WHEEL"

python - "$URL" "$WHEEL" <<'PY'
import sys
from urllib.request import urlretrieve

url, filename = sys.argv[1], sys.argv[2]
print("Downloading:", url)
urlretrieve(url, filename)
print("Saved:", filename)
PY

python -m pip install --no-cache-dir "./$WHEEL"
python -m pip install evo2==0.6.0 triton==3.3.1 vtx==1.1.0
python -m pip check

python -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
python -c 'from evo2 import Evo2; print("Evo2 import OK")'

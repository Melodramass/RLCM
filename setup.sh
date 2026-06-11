#!/usr/bin/env bash
# Environment setup for RLCM. Creates a fresh conda env and installs the
# full training/eval stack from scratch.
#
# Usage:
#   bash setup.sh                 # creates conda env "rlcm"
#   ENV_NAME=myenv bash setup.sh  # custom env name
set -e

ENV_NAME="${ENV_NAME:-rlcm}"

# 0. Fresh conda env (Python 3.10).
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "error: conda env '$ENV_NAME' already exists." >&2
    echo "Remove it (conda env remove -n $ENV_NAME) or pick another name (ENV_NAME=... bash setup.sh)." >&2
    exit 1
fi
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

# 1. Inference engine used for rollouts. Pins the rest of the heavy stack
#    (torch 2.9.0, CUDA 12.8 wheels).
pip install vllm==0.11.2

# 2. flash-attn, installed against the torch that vllm pinned.
pip install ninja packaging
pip install flash-attn==2.8.3 --no-build-isolation

# 3. The vendored verl fork (training framework).
pip install -e ./verl

# 4. The rlcm package (data utilities, probe model).
pip install -e .

# 5. Pinned versions known to work together (verl 0.8.0.dev requires
#    tensordict>=0.8,<=0.10 and latex2sympy2 requires antlr4 4.9.3).
pip install \
    transformers==4.57.6 \
    tensordict==0.10.0 \
    "ray[default]==2.53.0" \
    antlr4-python3-runtime==4.9.3 \
    pytest

# 6. Sanity check.
python -c "import torch, vllm, flash_attn, verl, rlcm, transformers, tensordict; \
print('torch', torch.__version__); print('vllm', vllm.__version__); \
print('flash_attn', flash_attn.__version__); print('verl', verl.__version__); \
print('transformers', transformers.__version__); print('tensordict', tensordict.__version__)"

echo "Done. Activate with: conda activate $ENV_NAME"

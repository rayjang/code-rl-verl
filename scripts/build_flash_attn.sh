#!/bin/bash
#SBATCH -p koni -w gpu48 -c 48 --mem=200G -t 03:00:00 -J fa_build
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/fa_build_%j.out
# Build flash-attn 2.8.3 against torch 2.11.0+cu130 for sm90 (H200) — no prebuilt wheel exists for torch 2.11.
set -uo pipefail
export CUDA_HOME=/apps/cuda/13.0.2 PATH=/apps/cuda/13.0.2/bin:$PATH LD_LIBRARY_PATH=/apps/cuda/13.0.2/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0" FLASH_ATTN_CUDA_ARCHS="90" MAX_JOBS=40 NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE
export TMPDIR=/tmp/r919a03_fa_$SLURM_JOB_ID; mkdir -p $TMPDIR
cd /scratch/r919a03/code_verl_test && source .venv/bin/activate && export UV_CACHE_DIR=/scratch/r919a03/.uv/cache
python -c "import torch; print(torch.__version__, torch.version.cuda)"; nvcc --version | tail -1
uv pip install --python .venv/bin/python wheel ninja packaging psutil setuptools 2>&1 | tail -1
time uv pip install --python .venv/bin/python --no-build-isolation --no-cache -v flash-attn==2.8.3 2>&1 | tail -40
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
rm -rf $TMPDIR

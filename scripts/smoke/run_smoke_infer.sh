#!/bin/bash
#SBATCH -p koni -w gpu48 --gres=gpu:1 -c 12 --mem=120G -t 01:00:00 -J smoke_inf
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/smoke_infer_%j.out
set -uo pipefail
cd /scratch/r919a03/code_verl_test
source .venv/bin/activate
export HF_HOME=/scratch/r919a03/huggingface HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
echo "host=$(hostname) CVD=$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,name,memory.used --format=csv
python scripts/smoke/smoke_infer_lora.py

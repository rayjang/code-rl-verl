#!/bin/bash
#SBATCH -p koni -w gpu48 --gres=gpu:1 -c 12 --mem=200G -t 02:00:00 -J smoke_q3
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/smoke_qwen3moe_%j.out
source .venv/bin/activate
export MODEL_PATH=/scratch/r919a03/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe HF_HOME=/scratch/r919a03/huggingface HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
echo "host=$(hostname) CVD=$CUDA_VISIBLE_DEVICES"; python scripts/smoke/smoke_qwen3moe.py

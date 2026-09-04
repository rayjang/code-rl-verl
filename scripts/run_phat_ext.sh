#!/bin/bash
# Qwen3 base-policy difficulty of the extended unit-test set (16,464 rows) on the 2 GPUs left free by Stage D.
#SBATCH -p koni -w gpu48 --gres=gpu:2 -c 20 --mem=200G -t 12:00:00 -J phat_ext
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/phat_ext_%j.out
set -uo pipefail
export ROOT=/scratch/r919a03/code_verl_test; cd $ROOT && source .venv/bin/activate
export PYTHONPATH=$ROOT HF_HOME=/scratch/r919a03/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export RL_INDEX_DIR=$ROOT/data/processed/index/curated_v3 RL_SIF_DIR=$ROOT/environments/sif
export VERIFIER_RUN_DIR=/tmp/r919a03_phat_ext_$SLURM_JOB_ID; mkdir -p $VERIFIER_RUN_DIR
export VERIFIER_CONFIG=$ROOT/configs/verifier/vf_v002.yaml REWARD_CONFIG=$ROOT/configs/reward/rw_v001_baseline.yaml
python scripts/measure_phat.py --parquet $ROOT/data/processed/ext_fixed.parquet --out $ROOT/results/phat/ext_q3_k8 --k 8 --gpus 2 --keep_text --concurrency 8 \
   --model /scratch/r919a03/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --max_model_len 8192 --max_tokens 4096 --max_num_seqs 64 2>&1 | grep -vE "^\s*$"
echo "end=$(date -Is)"; rm -rf $VERIFIER_RUN_DIR

#!/bin/bash
# Base-policy difficulty of Qwen3-30B-A3B on curated_v3pre (all SWE buckets incl. 64k/128k) — validation, test, train.
#SBATCH -p koni -w gpu48 --gres=gpu:6 -c 60 --mem=700G -t 12:00:00 -J phat_q3
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/phat_q3_%j.out
set -uo pipefail
export ROOT=/scratch/r919a03/code_verl_test; cd $ROOT && source .venv/bin/activate
V=${DATASET_VERSION:-curated_v3pre}; K=${K:-8}
export PYTHONPATH=$ROOT HF_HOME=/scratch/r919a03/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export RL_INDEX_DIR=$ROOT/data/processed/index/$V RL_SIF_DIR=$ROOT/environments/sif
export VERIFIER_RUN_DIR=/tmp/r919a03_phat_$SLURM_JOB_ID; mkdir -p $VERIFIER_RUN_DIR
export VERIFIER_CONFIG=$ROOT/configs/verifier/vf_v002.yaml REWARD_CONFIG=$ROOT/configs/reward/rw_v001_baseline.yaml
echo "host=$(hostname) CVD=$CUDA_VISIBLE_DEVICES start=$(date -Is) version=$V k=$K git=$(git rev-parse --short HEAD)"
for split in validation test train; do
  OUT=$ROOT/results/phat/${V}_${split}_q3_k$K; [ -f $OUT/summary.json ] && { echo "skip $split"; continue; }
  echo "=== $split $(date -Is) ==="
  python scripts/measure_phat.py --parquet $ROOT/data/curated/$V/$split.parquet --out $OUT --k $K --gpus 6 --keep_text --concurrency 10 \
      --model /scratch/r919a03/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe/ --max_model_len 135168 --max_tokens 4096 --max_num_seqs 32 2>&1 | grep -vE "^\s*$"
done
echo "end=$(date -Is)"; rm -rf $VERIFIER_RUN_DIR

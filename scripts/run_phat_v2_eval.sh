#!/bin/bash
#SBATCH -p koni -w gpu48 --gres=gpu:6 -c 60 --mem=600G -t 02:00:00 -J phat_v2
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/phat_v2_%j.out
set -uo pipefail
export ROOT=/scratch/r919a03/code_verl_test; cd $ROOT && source .venv/bin/activate
export PYTHONPATH=$ROOT HF_HOME=/scratch/r919a03/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export RL_INDEX_DIR=$ROOT/data/processed/index/curated_v2 RL_SIF_DIR=$ROOT/environments/sif
export VERIFIER_RUN_DIR=/tmp/r919a03_phat_$SLURM_JOB_ID; mkdir -p $VERIFIER_RUN_DIR
export VERIFIER_CONFIG=$ROOT/configs/verifier/vf_v002.yaml REWARD_CONFIG=$ROOT/configs/reward/rw_v001_baseline.yaml
MODEL=${MODEL_PATH:-}; TAG=${TAG:-base}; K=${K:-8}
for split in validation test; do
  OUT=$ROOT/results/phat/curated_v2_${split}_${TAG}_k${K}; [ -f $OUT/summary.json ] && continue
  python scripts/measure_phat.py --parquet $ROOT/data/curated/curated_v2/$split.parquet --out $OUT --k $K --gpus 6 --keep_text --concurrency 10 ${MODEL:+--model $MODEL} ${LORA:+--lora $LORA} 2>&1 | grep -vE "^\s*$"
done
echo "end=$(date -Is)"; rm -rf $VERIFIER_RUN_DIR

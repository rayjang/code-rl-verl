#!/bin/bash
# Measure empirical success probability of the BASE policy on curated splits (k samples each) — Stage B/C.
#SBATCH -p koni -w gpu48 --gres=gpu:4 -c 40 --mem=400G -t 08:00:00 -J phat_all
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/phat_all_%j.out
set -uo pipefail
export ROOT=/scratch/r919a03/code_verl_test
cd $ROOT && source .venv/bin/activate
NG=${NGPU:-4}; V=${DATASET_VERSION:-curated_v1}; K=${K:-8}
export PYTHONPATH=$ROOT HF_HOME=/scratch/r919a03/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export RL_INDEX_DIR=$ROOT/data/processed/index/$V RL_SIF_DIR=$ROOT/environments/sif
export VERIFIER_RUN_DIR=/tmp/r919a03_phat_$SLURM_JOB_ID; mkdir -p $VERIFIER_RUN_DIR
export VERIFIER_CONFIG=$ROOT/configs/verifier/vf_v002.yaml REWARD_CONFIG=$ROOT/configs/reward/rw_v001_baseline.yaml
echo "host=$(hostname) CVD=$CUDA_VISIBLE_DEVICES start=$(date -Is) version=$V k=$K ngpu=$NG git=$(git rev-parse --short HEAD)"
for split in validation test train; do
  OUT=$ROOT/results/phat/${V}_${split}_k${K}
  [ -f $OUT/summary.json ] && { echo "skip $split (done)"; continue; }
  echo "=== $split $(date -Is) ==="
  python scripts/measure_phat.py --parquet $ROOT/data/curated/$V/$split.parquet --out $OUT --k $K --gpus $NG --keep_text --concurrency 10 2>&1 | grep -vE "^\s*$"
done
echo "end=$(date -Is)"; rm -rf $VERIFIER_RUN_DIR

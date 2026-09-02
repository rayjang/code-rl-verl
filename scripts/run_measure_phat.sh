#!/bin/bash
# Usage: sbatch scripts/run_measure_phat.sh <parquet> <out_dir> <k> [extra args...]
#SBATCH -p koni -w gpu48 --gres=gpu:6 -c 48 --mem=600G -t 06:00:00 -J phat
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/phat_%j.out
set -uo pipefail
export ROOT=/scratch/r919a03/code_verl_test
cd $ROOT && source .venv/bin/activate
export PYTHONPATH=$ROOT HF_HOME=/scratch/r919a03/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export RL_INDEX_DIR=${RL_INDEX_DIR:-$ROOT/data/processed/index/curated_v1} RL_SIF_DIR=$ROOT/environments/sif
export VERIFIER_RUN_DIR=/tmp/r919a03_phat_$SLURM_JOB_ID; mkdir -p $VERIFIER_RUN_DIR
export VERIFIER_CONFIG=${VERIFIER_CONFIG:-$ROOT/configs/verifier/vf_v002.yaml} REWARD_CONFIG=${REWARD_CONFIG:-$ROOT/configs/reward/rw_v001_baseline.yaml}
PARQUET=$1; OUT=$2; K=${3:-8}; shift 3 || true
echo "host=$(hostname) CVD=$CUDA_VISIBLE_DEVICES start=$(date -Is) parquet=$PARQUET out=$OUT k=$K extra=$*"
python scripts/measure_phat.py --parquet $PARQUET --out $OUT --k $K --gpus 6 --keep_text "$@"
echo "end=$(date -Is)"; rm -rf $VERIFIER_RUN_DIR

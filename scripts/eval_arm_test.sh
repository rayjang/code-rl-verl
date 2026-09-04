#!/bin/bash
# Final held-out evaluation of a trained arm: export adapter (if needed) -> k samples on curated_v2 validation+test.
# Usage: sbatch scripts/eval_arm_test.sh <exp_id> [k]
#SBATCH -p koni -w gpu48 --gres=gpu:4 -c 40 --mem=480G -t 08:00:00 -J eval_arm
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/eval_arm_%j.out
set -uo pipefail
ROOT=/scratch/r919a03/code_verl_test; cd $ROOT && source .venv/bin/activate
EXP=$1; K=${2:-8}
if [ ! -f checkpoints/$EXP/lora_adapter/adapter_model.safetensors ]; then
  STEP=$(cat checkpoints/$EXP/latest_checkpointed_iteration.txt); TMP=/tmp/r919a03_merge_$SLURM_JOB_ID; mkdir -p $TMP
  python -m verl.model_merger merge --backend fsdp --local_dir checkpoints/$EXP/global_step_$STEP/actor --target_dir $TMP 2>&1 | tail -3
  cp -r $TMP/lora_adapter checkpoints/$EXP/lora_adapter && rm -rf $TMP
fi
export LORA=$ROOT/checkpoints/$EXP/lora_adapter TAG=$EXP K=$K NGPU=${NGPU:-4} DATASET_VERSION=${DATASET_VERSION:-curated_v3}
export MODEL_PATH=$(grep -oE "export MODEL_PATH=\S+" experiments/$EXP/env.sh | cut -d= -f2)
bash scripts/run_phat_v2_eval.sh

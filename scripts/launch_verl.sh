#!/bin/bash
# Launch one verl experiment on gpu48 (6 GPUs). Usage: sbatch scripts/launch_verl.sh <exp_dir>
#SBATCH -p koni -w gpu48 --gres=gpu:6 -c 72 --mem=700G -t 24:00:00 -J verl
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/verl_%j.out
set -uo pipefail
export ROOT=/scratch/r919a03/code_verl_test
EXP_DIR=${1:?exp_dir}
cd $ROOT && source .venv/bin/activate && source $EXP_DIR/env.sh
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HF_HOME=/scratch/r919a03/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export RL_INDEX_DIR=$ROOT/sources/rl_code_v1/data/index RL_SIF_DIR=$ROOT/environments/sif
export VERIFIER_RUN_DIR=/tmp/r919a03_verifier_${SLURM_JOB_ID} RUSCA_STEP_FILE=/tmp/r919a03_rusca_step_${SLURM_JOB_ID}
export VERIFIER_LOG_JSONL=$EXP_DIR/instance_log.jsonl
export RAY_TMPDIR=/tmp/r919a03_ray_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/r919a03_inductor
export NCCL_DEBUG=WARN
mkdir -p $VERIFIER_RUN_DIR $RAY_TMPDIR $EXP_DIR
echo "host=$(hostname) job=$SLURM_JOB_ID CVD=$CUDA_VISIBLE_DEVICES start=$(date -Is) git=$(git rev-parse --short HEAD)" | tee $EXP_DIR/run_info.txt
nvidia-smi --query-gpu=index,name,memory.used --format=csv | tee -a $EXP_DIR/run_info.txt
python -c "import torch,vllm,verl,transformers,peft,flash_attn; print('torch',torch.__version__,'vllm',vllm.__version__,'verl',verl.__version__,'transformers',transformers.__version__,'peft',peft.__version__,'flash_attn',flash_attn.__version__)" | tee -a $EXP_DIR/run_info.txt
# expand $ROOT/$MODEL_PATH/$EXP_NAME inside overrides
mapfile -t OV < <(envsubst '$ROOT $MODEL_PATH $EXP_NAME' < $EXP_DIR/overrides.txt | grep -vE '^\s*(#|$)')
printf '%s\n' "${OV[@]}" > $EXP_DIR/overrides.resolved.txt
echo "START_TRAIN $(date -Is)"
python -m rl.main_ppo_code "${OV[@]}" 2>&1 | tee $EXP_DIR/train.log
rc=${PIPESTATUS[0]}
echo "END_TRAIN rc=$rc $(date -Is)" | tee -a $EXP_DIR/run_info.txt
ray stop --force >/dev/null 2>&1 || true
rm -rf $VERIFIER_RUN_DIR $RAY_TMPDIR
exit $rc

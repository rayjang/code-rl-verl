#!/bin/bash
# Launch one verl experiment on gpu48 (6 GPUs). Usage: sbatch scripts/launch_verl.sh <exp_dir>
#SBATCH -p koni -w gpu48 --gres=gpu:6 -c 48 --mem=720G -t 24:00:00 -J verl
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/verl_%j.out
set -uo pipefail
export ROOT=/scratch/r919a03/code_verl_test
export NGPU=${NGPU:-6}
EXP_DIR=${1:?exp_dir}
cd $ROOT && source .venv/bin/activate && source $EXP_DIR/env.sh
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HF_HOME=/scratch/r919a03/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export RL_INDEX_DIR=$ROOT/data/processed/index/${DATASET_VERSION:-curated_v2} RL_SIF_DIR=$ROOT/environments/sif
export VERIFIER_RUN_DIR=/tmp/r919a03_verifier_${SLURM_JOB_ID} RUSCA_STEP_FILE=/tmp/r919a03_rusca_step_${SLURM_JOB_ID}
export VERIFIER_LOG_JSONL=$EXP_DIR/instance_log.jsonl
export RAY_TMPDIR=/tmp/r919a03_ray_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/r919a03_inductor
export NCCL_DEBUG=WARN
mkdir -p $VERIFIER_RUN_DIR $RAY_TMPDIR $EXP_DIR
ray stop --force >/dev/null 2>&1 || true   # clear any stale Ray session from a previous job on this node
echo "host=$(hostname) job=$SLURM_JOB_ID CVD=$CUDA_VISIBLE_DEVICES start=$(date -Is) git=$(git rev-parse --short HEAD)" | tee $EXP_DIR/run_info.txt
nvidia-smi --query-gpu=index,name,memory.used --format=csv | tee -a $EXP_DIR/run_info.txt
python -c "import torch,vllm,verl,transformers,peft,flash_attn; print('torch',torch.__version__,'vllm',vllm.__version__,'verl',verl.__version__,'transformers',transformers.__version__,'peft',peft.__version__,'flash_attn',flash_attn.__version__)" | tee -a $EXP_DIR/run_info.txt
# expand $ROOT/$MODEL_PATH/$EXP_NAME inside overrides
mapfile -t OV < <(envsubst '$ROOT $MODEL_PATH $EXP_NAME $NGPU' < $EXP_DIR/overrides.txt | grep -vE '^\s*(#|$)')
printf '%s\n' "${OV[@]}" > $EXP_DIR/overrides.resolved.txt
# host-memory trace (diagnose OOM kills): top RSS processes every 20s
( while true; do echo "== $(date -Is)"; free -g | sed -n 2p; ps -eo pid,rss,comm --sort=-rss | head -14; sleep 20; done ) > $EXP_DIR/mem_trace.log 2>&1 &
MEMTRACE_PID=$!
# Rollout stall watchdog. An async vLLM rollout occasionally wedges with a couple of requests stuck
# ("pending: 0, running: 2, finished: 22" in exp_Q003_f2p_ladder, 2026-09-05) and then burns the whole
# wall clock: the progress line keeps printing every 60 s, so the log mtime is NOT a stall signal --
# the *content* of the last progress line is. A healthy run repeats one such line at most ~13 times
# (one long 128k rollout finishing last); anything past STALL_MIN is wedged. Kill it so the run exits
# non-zero and the autoresearch loop retries, instead of losing a 10 h slot to a hang.
STALL_MIN=${STALL_MIN:-45}
( same=0; last=""
  while sleep 60; do
    cur=$(grep -oE "pending: [0-9]+, running: [0-9]+, finished: [0-9]+" $EXP_DIR/train.log 2>/dev/null | tail -1)
    [[ -z "$cur" ]] && { same=0; continue; }
    if [[ "$cur" == "$last" ]]; then same=$((same+1)); else same=0; last="$cur"; fi
    if (( same >= STALL_MIN )); then
      echo "STALL_WATCHDOG $(date -Is): no rollout progress for ${STALL_MIN} min at [$cur] -- killing" \
        | tee -a $EXP_DIR/run_info.txt $EXP_DIR/train.log
      pkill -f "rl.main_ppo_code" 2>/dev/null
      break
    fi
  done ) > $EXP_DIR/stall_watchdog.log 2>&1 &
WATCHDOG_PID=$!
echo "START_TRAIN $(date -Is) NGPU=$NGPU stall_watchdog=${STALL_MIN}min"
python -m rl.main_ppo_code "${OV[@]}" 2>&1 | tee $EXP_DIR/train.log
rc=${PIPESTATUS[0]}
kill $MEMTRACE_PID $WATCHDOG_PID 2>/dev/null
echo "END_TRAIN rc=$rc $(date -Is)" | tee -a $EXP_DIR/run_info.txt
ray stop --force >/dev/null 2>&1 || true
rm -rf $VERIFIER_RUN_DIR $RAY_TMPDIR
exit $rc

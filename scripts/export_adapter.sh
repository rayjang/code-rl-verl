#!/bin/bash
# Export the LoRA adapter (PEFT format) from a verl FSDP checkpoint. Usage: sbatch scripts/export_adapter.sh <exp_id> [step]
#SBATCH -p koni -w gpu48 -c 8 --mem=96G -t 01:00:00 -J export_lora
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/export_%j.out
set -uo pipefail
ROOT=/scratch/r919a03/code_verl_test; cd $ROOT && source .venv/bin/activate
EXP=$1; STEP=${2:-$(cat checkpoints/$EXP/latest_checkpointed_iteration.txt)}
SRC=checkpoints/$EXP/global_step_$STEP/actor; TMP=/tmp/r919a03_merge_$SLURM_JOB_ID; mkdir -p $TMP
echo "exp=$EXP step=$STEP src=$SRC"
python -m verl.model_merger merge --backend fsdp --local_dir $SRC --target_dir $TMP 2>&1 | tail -15
ls -la $TMP $TMP/lora_adapter 2>/dev/null | head -20
if [ -d $TMP/lora_adapter ]; then
  rm -rf checkpoints/$EXP/lora_adapter && cp -r $TMP/lora_adapter checkpoints/$EXP/lora_adapter && echo "ADAPTER_OK checkpoints/$EXP/lora_adapter"
else
  echo "NO_ADAPTER_DIR"; ls $TMP
fi
rm -rf $TMP

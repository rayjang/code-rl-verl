#!/bin/bash
#SBATCH -p koni -w gpu48 -c 32 --mem=128G -t 08:00:00 -J val_env
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/val_env_%j.out
set -uo pipefail
source .venv/bin/activate
export VERIFIER_RUN_DIR=/tmp/r919a03_verifier_run_$SLURM_JOB_ID
mkdir -p $VERIFIER_RUN_DIR
echo "host=$(hostname) start=$(date -Is)"
python scripts/validate_environments.py --track swe --workers 16 2>&1 | grep -v "^$"
# ut already validated (800/800 ok)
echo "end=$(date -Is)"; rm -rf $VERIFIER_RUN_DIR

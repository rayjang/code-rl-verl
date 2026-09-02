#!/bin/bash
#SBATCH -p koni -w gpu48 -c 24 --mem=64G -t 04:00:00 -J val_ut
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/val_ut_%j.out
set -uo pipefail
source .venv/bin/activate
export VERIFIER_RUN_DIR=/tmp/r919a03_verifier_ut_$SLURM_JOB_ID; mkdir -p $VERIFIER_RUN_DIR
echo "host=$(hostname) start=$(date -Is)"
python scripts/validate_environments.py --track ut --workers 22 2>&1 | grep -v "^$"
echo "end=$(date -Is)"; rm -rf $VERIFIER_RUN_DIR

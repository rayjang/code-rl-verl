#!/bin/bash
#SBATCH -p koni -w gpu48 -c 8 --mem=32G -t 00:40:00 -J dbg_swe
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/dbg_swe_%j.out
source .venv/bin/activate; export VERIFIER_RUN_DIR=/tmp/r919a03_dbg_$SLURM_JOB_ID; mkdir -p $VERIFIER_RUN_DIR
python scripts/debug_swe_repo.py; rm -rf $VERIFIER_RUN_DIR

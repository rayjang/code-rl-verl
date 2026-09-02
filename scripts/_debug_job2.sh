#!/bin/bash
#SBATCH -p koni -w gpu48 -c 4 --mem=24G -t 00:30:00 -J dbg_swe2
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/dbg_swe2_%j.out
source .venv/bin/activate; export VERIFIER_RUN_DIR=/tmp/r919a03_dbg2_$SLURM_JOB_ID; mkdir -p $VERIFIER_RUN_DIR
python scripts/debug_swe_repo2.py; rm -rf $VERIFIER_RUN_DIR

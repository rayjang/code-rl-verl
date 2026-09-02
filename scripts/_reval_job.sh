#!/bin/bash
#SBATCH -p koni -w gpu48 -c 8 --mem=32G -t 00:30:00 -J reval
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/reval_%j.out
source .venv/bin/activate; export VERIFIER_RUN_DIR=/tmp/r919a03_reval_$SLURM_JOB_ID; mkdir -p $VERIFIER_RUN_DIR
python scripts/validate_environments.py --track swe --workers 8 2>&1 | grep -v "^$"
python scripts/build_dataset.py --version curated_v1 --seed 0 2>&1 | grep -v Warning | head -30
rm -rf $VERIFIER_RUN_DIR

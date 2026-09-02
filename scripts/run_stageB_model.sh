#!/bin/bash
# Stage B on real policy samples: parser arms over results/phat/<version>_<split>_k8/responses.jsonl (no execution)
#SBATCH -p koni -w gpu48 -c 4 --mem=16G -t 01:00:00 -J stageB
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH -o logs/stageB_%j.out
set -uo pipefail
source .venv/bin/activate
V=${DATASET_VERSION:-curated_v1}
for split in validation train; do
  R=results/phat/${V}_${split}_k8/responses.jsonl
  [ -f $R ] || { echo "missing $R"; continue; }
  python scripts/verifier_ablation.py --tag model_${split} --n_synthetic 50 --responses $R
done

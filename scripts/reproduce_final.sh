#!/bin/bash
# End-to-end reproduction on gpu48 (KISTI Neuron). Every step is idempotent / resumable.
# Prerequisites: sources/ extracted from the two archives, Qwen1.5-MoE snapshot in $HF_HOME, .venv built (see docs/source_inventory.md §3).
set -euo pipefail
ROOT=/scratch/r919a03/code_verl_test; cd $ROOT
export NGPU=${NGPU:-4}
echo "[1/8] unit tests";               source .venv/bin/activate && python -m pytest tests/verifier -q
echo "[2/8] images";                   [ -f environments/sif/swesmith.x86_64.suor_1776_funcy.207a7810.sif ] || sbatch --wait scripts/pull_images.sh
echo "[3/8] unit-test sandbox";        [ -x environments/ut_venv/bin/python ] || sbatch --wait scripts/build_ut_sandbox.sh
echo "[4/8] environment validation";   [ -f environments/manifests/validation_swe.jsonl ] || sbatch --wait scripts/run_validate_envs.sh
                                       [ -f environments/manifests/validation_ut.jsonl ] || sbatch --wait scripts/run_validate_ut_full.sh
echo "[5/8] curated_v1";               python scripts/build_dataset.py --version curated_v1 --seed 0
echo "[6/8] base-policy difficulty";   [ -f results/phat/curated_v1_train_k8/summary.json ] || sbatch --wait scripts/run_phat_all.sh
echo "[7/8] curated_v2 + Stage D";     python scripts/build_dataset.py --version curated_v2 --seed 0 --phat results/phat/curated_v1_train_k8/phat.jsonl --band 0.05,0.95
                                       python scripts/autoresearch.py --plan configs/plans/stageD.yaml
echo "[8/8] reports";                  python scripts/report_experiments.py && python scripts/report_dataset_quality.py --version curated_v2 && python scripts/build_excel.py

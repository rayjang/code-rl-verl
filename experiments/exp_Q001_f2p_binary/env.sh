export EXP_NAME=exp_Q001_f2p_binary
export MODEL_PATH=/scratch/r919a03/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
export DATASET_VERSION=curated_v3
export REWARD_CONFIG=$ROOT/configs/reward/rw_v002_f2p_binary.yaml
export VERIFIER_CONFIG=$ROOT/configs/verifier/vf_v002.yaml
export RUBRIC_CONFIG=$ROOT/rubrics/rubric_v001_code_hint.yaml
export VERIFIER_CONCURRENCY=6

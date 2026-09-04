#!/bin/bash
# wait for the Qwen3 p-hat job, build curated_v3, start the Stage D (Qwen3) loop
cd /scratch/r919a03/code_verl_test; source .venv/bin/activate
until grep -q "^end=" logs/phat_q3_903380.out 2>/dev/null; do sleep 120; done
echo "phat_q3 finished $(date -Is)"
python scripts/finalize_v3.py 2>&1 | grep -vE "Warning" | tail -20
python -c "
import json; m=json.load(open('data/curated/curated_v3/dataset_manifest.json')); print(json.dumps(m['counts'])); print(m['exclusions'])"
git add data/curated/curated_v3/dataset_manifest.json data/curated/curated_v3/instance_audit.csv data/curated/curated_v3/prompt_fix_decision.json docs/dataset_quality_report.md results/phat/curated_v3pre_*/summary.json results/phat/curated_v3pre_*/phat.jsonl 2>/dev/null
git commit -qm "data: curated_v3 (Qwen3 base-policy band filter, all SWE buckets)" && echo committed
nohup python scripts/autoresearch.py --plan configs/plans/stageD_q3.yaml > logs/autoresearch_stageD_q3.log 2>&1 &
echo "loop started pid $!"

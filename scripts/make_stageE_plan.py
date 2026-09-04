#!/usr/bin/env python3
"""Fill configs/plans/stageE.yaml with the Stage-D winner (and runner-up) as long-run arms."""
import json, os, sys, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scripts.experiment import load_registry  # noqa: E402
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--state", default=f"{ROOT}/experiments/plan_state_stageD_q3.json")
ap.add_argument("--plan", default=f"{ROOT}/configs/plans/stageE_q3.yaml"); ap.add_argument("--baseline", default="exp_Q000_baseline")
a = ap.parse_args()
reg = load_registry(); state = json.load(open(a.state))
plan = yaml.safe_load(open(a.plan))
best = state["best"]; r = reg[best]
arms = []
if best != a.baseline:
    arm = {"id": "exp_E001_winner_long", "hypothesis": f"Stage-D winner {best} keeps its advantage over 100 steps", "changes": r.get("changed_variables", ""),
           "reward": r.get("reward_config"), "verifier": r.get("verifier_config")}
    if r.get("rubric_config"):
        arm["rubric"] = r["rubric_config"]
    extra = [o for o in r.get("overrides", []) if o.startswith(("algorithm.adv_estimator=grpo_ddca", "+algorithm.ddca_beta", "+actor_rollout_ref.rollout.custom.rusca.enable=False"))]
    if extra:
        arm["set"] = extra
    arms.append(arm)
plan["hypotheses"] = arms
yaml.safe_dump(plan, open(a.plan, "w"), sort_keys=False, allow_unicode=True)
print(json.dumps(plan, indent=1, ensure_ascii=False))

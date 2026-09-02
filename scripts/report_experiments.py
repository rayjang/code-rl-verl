#!/usr/bin/env python3
"""Render docs/experiment_report.md and docs/ablation_report.md tables from the experiment registry + metrics."""
import json, os, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scripts.experiment import load_registry  # noqa: E402


def val_keys(s):
    return {k: v for k, v in (s.get("val") or {}).items()}


def main():
    reg = load_registry()
    rows = []
    for i, r in reg.items():
        s = r.get("summary") or {}
        v = val_keys(s)
        pick = lambda sub: (round(v[sub], 4) if sub in v else next((round(v[k], 4) for k in v if sub in k), None))
        rows.append({"id": i, "parent": r.get("parent"), "stage": r.get("stage"), "reward": r.get("reward_version"), "verifier": r.get("verifier_version"),
                     "rubric": r.get("rubric_version"), "git": r.get("git_commit"), "seed": r.get("seed"), "job": r.get("slurm_job"), "status": r.get("status"),
                     "steps": s.get("n_steps"), "train_reward": None if s.get("final_reward") is None else round(s["final_reward"], 4),
                     "val_score": pick("val_score"), "val_resolved": pick("val_resolved"), "val_f2p": pick("val_f2p"), "val_p2p": pick("val_p2p"),
                     "val_apply": pick("val_apply"), "val_format": pick("val_format"), "val_resolved_swe": pick("val_resolved_swe"), "val_resolved_ut": pick("val_resolved_ut"), "kl": None if s.get("final_kl") is None else round(s["final_kl"], 5),
                     "entropy": None if s.get("final_entropy") is None else round(s["final_entropy"], 4), "max_grad": None if s.get("max_grad_norm") is None else round(s["max_grad_norm"], 3),
                     "decision": r.get("decision"), "reason": r.get("decision_reason"), "hypothesis": r.get("hypothesis"), "changes": r.get("changed_variables")})
    df = pd.DataFrame(rows)
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    df.to_csv(f"{ROOT}/results/experiments_table.csv", index=False)
    cols = ["id", "stage", "reward", "verifier", "rubric", "steps", "train_reward", "val_score", "val_resolved", "val_resolved_swe", "val_resolved_ut", "val_f2p", "val_p2p", "val_apply", "kl", "entropy", "max_grad", "decision"]
    md = ["# Experiment report\n", "Registry: `experiments/registry.jsonl`; per-experiment folders hold `overrides.resolved.txt`, `train.log`, `instance_log.jsonl`, `metrics.json`.\n",
          "## All experiments\n", df[cols].to_markdown(index=False) if len(df) else "(none yet)", "\n## Hypotheses and decisions\n"]
    for _, r in df.iterrows():
        md.append(f"* **{r['id']}** (parent {r['parent']}): {r['hypothesis']} — changed: `{r['changes']}` → **{r['decision']}**: {r['reason']}")
    open(f"{ROOT}/docs/experiment_report.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(df[cols].to_string())


if __name__ == "__main__":
    main()

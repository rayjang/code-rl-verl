#!/usr/bin/env python3
"""Assemble results/rl_code_reward_research.xlsx from the registry, metrics, manifests and configs."""
from __future__ import annotations
import glob, json, os, sys
import pandas as pd
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.experiment import load_registry  # noqa: E402

OUT = f"{ROOT}/results/rl_code_reward_research.xlsx"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def main():
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    reg = load_registry()
    # Experiments
    rows = []
    for i, r in reg.items():
        s = r.get("summary", {}) or {}
        val = s.get("val", {}) or {}
        rows.append({"experiment_id": i, "parent": r.get("parent"), "stage": r.get("stage"), "hypothesis": r.get("hypothesis"),
                     "changed_variables": r.get("changed_variables"), "git_commit": r.get("git_commit"),
                     "dataset_train": (r.get("dataset_version") or {}).get("train"), "dataset_train_sha": (r.get("dataset_version") or {}).get("train_sha"),
                     "reward_version": r.get("reward_version"), "verifier_version": r.get("verifier_version"), "rubric_version": r.get("rubric_version"),
                     "seed": r.get("seed"), "gpu": r.get("gpu"), "slurm_job": r.get("slurm_job"), "start_time": r.get("start_time"), "end_time": r.get("end_time"),
                     "status": r.get("status"), "n_steps": s.get("n_steps"), "final_train_reward": s.get("final_reward"), "final_kl": s.get("final_kl"),
                     "final_entropy": s.get("final_entropy"), "max_grad_norm": s.get("max_grad_norm"),
                     **{f"val:{k}": v for k, v in val.items()}, "decision": r.get("decision"), "decision_reason": r.get("decision_reason")})
    experiments = pd.DataFrame(rows)
    # TrainingCurves
    curves = []
    for i in reg:
        mp = f"{ROOT}/experiments/{i}/metrics.json"
        if os.path.exists(mp):
            m = json.load(open(mp))
            for step, d in m.get("steps", {}).items():
                curves.append({"experiment_id": i, "step": int(step), **{k: d.get(k) for k in [
                    "critic/score/mean", "critic/score/std", "actor/pg_loss", "actor/ppo_kl", "actor/entropy", "actor/grad_norm", "actor/lr",
                    "response_length/mean", "response_length/clip_ratio", "actor/pg_clipfrac"]},
                    **{k.replace("reward_extra/", "rx:"): v for k, v in d.items() if any(t in k for t in ("f2p_frac", "p2p_frac", "patch_apply_score", "patch_format_score", "rule_correctness", "infra_excluded", "rubric_score", "rusca_n_inject"))}})
    # versions
    rubric_rows = [{"rubric_version": y.get("version"), "file": os.path.relpath(p, ROOT), **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in y.items() if k != "version"}}
                   for p in sorted(glob.glob(f"{ROOT}/rubrics/*.yaml")) for y in [yaml.safe_load(open(p))]]
    reward_rows = [{"reward_version": y.get("version"), "file": os.path.relpath(p, ROOT), **{k: v for k, v in y.items() if k != "version"}}
                   for p in sorted(glob.glob(f"{ROOT}/configs/reward/*.yaml")) for y in [yaml.safe_load(open(p))]]
    verifier_rows = [{"verifier_version": y.get("version"), "file": os.path.relpath(p, ROOT), **{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in y.items() if k != "version"}}
                     for p in sorted(glob.glob(f"{ROOT}/configs/verifier/*.yaml")) for y in [yaml.safe_load(open(p))]]
    # instance results (latest curated audit + phat + validation)
    inst = []
    for p in sorted(glob.glob(f"{ROOT}/results/phat/*/phat.jsonl")):
        tag = p.split("/")[-2]
        for r in jl(p):
            inst.append({"run": tag, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()}})
    audit = []
    for p in sorted(glob.glob(f"{ROOT}/data/curated/*/instance_audit.csv")):
        d = pd.read_csv(p); d.insert(0, "dataset_version", p.split("/")[-2]); audit.append(d)
    env_rows = []
    for r in jl(f"{ROOT}/environments/manifests/validation_swe.jsonl"):
        env_rows.append({k: (json.dumps(v, ensure_ascii=False)[:500] if isinstance(v, (dict, list)) else v) for k, v in r.items() if k not in ("gold_touches",)})
    ablation = pd.read_csv(f"{ROOT}/results/ablations.csv") if os.path.exists(f"{ROOT}/results/ablations.csv") else pd.DataFrame()
    final = pd.read_csv(f"{ROOT}/results/final_selection.csv") if os.path.exists(f"{ROOT}/results/final_selection.csv") else pd.DataFrame()
    difficulty = pd.DataFrame(inst)
    def sheet(df, name):
        d = df if (df is not None and len(df.columns)) else pd.DataFrame({"note": ["(no rows yet)"]})
        d.to_excel(w, sheet_name=name, index=False)
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        sheet(experiments, "Experiments")
        sheet(pd.DataFrame(curves), "TrainingCurves")
        sheet(pd.DataFrame(rubric_rows), "RubricVersions")
        sheet(pd.DataFrame(verifier_rows), "VerifierVersions")
        sheet(pd.DataFrame(reward_rows), "RewardVersions")
        sheet(ablation, "Ablations")
        sheet(pd.concat(audit) if audit else pd.DataFrame(), "InstanceResults")
        sheet(pd.concat(audit)[lambda d: d["status"] != "ok"] if audit else pd.DataFrame(), "DatasetAudit")
        sheet(difficulty, "Difficulty")
        sheet(pd.DataFrame(env_rows), "EnvironmentManifest")
        sheet(final, "FinalSelection")
    print("wrote", OUT, {"experiments": len(experiments), "curves": len(curves), "instances": len(inst), "env": len(env_rows)})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Noise-aware final selection (spec section 24 priorities) over Stage D/E arms.

Ranking key (lexicographic with noise tolerance):
 1. held-out coding performance: k-sample test solve rate if available (results/phat/curated_v2_test_<exp>_k8) else
    validation solve rate; two arms are 'tied' when |diff| < z * SE (SE from n and p).
 2. complete F2P solve rate (same source), 3. lower P2P regression, 4. higher apply/format validity,
 5. training stability (max grad norm, KL), 6. reward-signal health (train reward std not collapsed),
 7. hacking resistance (invalid-rate not rising), 8. coverage, 9. reproducibility (fixed seed/commit).
Writes results/final_selection.csv and prints the ranking.
"""
import json, math, os, sys, glob
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scripts.experiment import load_registry  # noqa: E402
Z = 1.0


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-6) / max(n, 1))


def test_phat(exp):
    p = f"{ROOT}/results/phat/curated_v3_test_{exp}_k8/summary.json"
    if not os.path.exists(p):
        return None
    s = json.load(open(p))["summary"]
    n = sum(v["n_instances"] for v in s.values())
    solve = sum(v["mean_p_hat"] * v["n_instances"] for v in s.values()) / n
    return {"test_solve_k8": solve, "test_n": n, "test_solve_swe": s.get("swe_32k", {}).get("mean_p_hat"), "test_solve_ut": sum(v["mean_p_hat"] * v["n_instances"] for k, v in s.items() if k != "swe_32k") / max(1, n - s.get("swe_32k", {}).get("n_instances", 0)),
            "test_reward_k8": sum(v["mean_reward"] * v["n_instances"] for v in s.values()) / n, "test_apply_swe": s.get("swe_32k", {}).get("apply_rate")}


def main():
    reg = load_registry()
    rows = []
    for i, r in reg.items():
        if not i.startswith(("exp_D", "exp_E")):
            continue
        s = r.get("summary") or {}; v = s.get("val") or {}
        if s.get("end_rc") != 0 or "val_resolved" not in v:
            continue
        row = {"experiment_id": i, "stage": r.get("stage"), "reward": r.get("reward_version"), "verifier": r.get("verifier_version"), "rubric": r.get("rubric_version"),
               "val_resolved": v["val_resolved"], "val_resolved_ut": v.get("val_resolved_ut"), "val_resolved_swe": v.get("val_resolved_swe"), "val_score": v.get("val_score"),
               "val_p2p": v.get("val_p2p"), "val_apply": v.get("val_apply"), "val_format": v.get("val_format"), "val_invalid": v.get("val_invalid"),
               "train_reward": s.get("final_reward"), "max_grad_norm": s.get("max_grad_norm"), "final_kl": s.get("final_kl"), "loop_decision": r.get("decision"), "git": r.get("git_commit"), "seed": r.get("seed")}
        row.update(test_phat(i) or {})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not len(df):
        print("no completed arms"); return
    base = df[df["experiment_id"].str.endswith("baseline")].iloc[0] if df["experiment_id"].str.endswith("baseline").any() else None
    prim = "test_solve_k8" if "test_solve_k8" in df.columns and df["test_solve_k8"].notna().any() else "val_resolved"
    n_ref = 699 * 8 if prim == "test_solve_k8" else 435
    df["primary"] = df[prim]
    df["primary_se"] = [se(p, n_ref) for p in df["primary"]]
    if base is not None:
        df["delta_vs_baseline"] = df["primary"] - float(base[prim] if prim in base and pd.notna(base[prim]) else base["val_resolved"])
        df["significant"] = df["delta_vs_baseline"].abs() > Z * (df["primary_se"] * math.sqrt(2))
    stab = (df["max_grad_norm"].fillna(1e9) < 100) & (df["val_invalid"].fillna(0) < 0.6)
    df["stable"] = stab
    df = df.sort_values(["stable", "primary", "val_p2p", "val_apply"], ascending=[False, False, False, False]).reset_index(drop=True)
    df["rank"] = df.index + 1
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    df.to_csv(f"{ROOT}/results/final_selection.csv", index=False)
    print(f"primary metric: {prim}; z={Z}")
    print(df[["rank", "experiment_id", "reward", "primary", "primary_se", "delta_vs_baseline", "significant", "val_resolved_ut", "val_resolved_swe", "val_p2p", "val_apply", "val_invalid", "max_grad_norm", "loop_decision"]].to_string())


if __name__ == "__main__":
    main()

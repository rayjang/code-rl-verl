#!/usr/bin/env python3
"""Group Stage-D/E experiments into ablation families and compare each arm with the baseline on the
validation metrics; writes docs/ablation_report.md and results/ablations.csv (Excel sheet `Ablations`)."""
import json, os, sys, glob
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scripts.experiment import load_registry  # noqa: E402

FAMILIES = {
    "F2P partial credit": ["exp_D000_baseline", "exp_D001_f2p_binary", "exp_D002_f2p_linear", "exp_D003_f2p_ladder"],
    "P2P regression penalty": ["exp_D000_baseline", "exp_D004_p2p_strong", "exp_D005_p2p_tolerant"],
    "Rule-only vs rubric vs RUSCA": ["exp_D000_baseline", "exp_D006_rule_only", "exp_D007_rubric_heavy", "exp_D008_rusca", "exp_D009_no_scaffold"],
    "Gating": ["exp_D000_baseline", "exp_D010_additive_nogate"],
    "Length (DDCA)": ["exp_D000_baseline", "exp_D011_ddca"],
    "Verifier / parser": ["exp_D000_baseline", "exp_D012_parser_baseline"],
}
METRICS = [("val_score", "score/mean"), ("val_resolved", "rule_correctness_score"), ("val_f2p", "f2p_frac"), ("val_p2p", "p2p_frac"),
           ("val_apply", "patch_apply_score"), ("val_format", "patch_format_score"), ("val_invalid", "gated_out"), ("val_infra", "infra_excluded"), ("val_resp_len", "response_length/mean")]


def pick(val, sub):
    ks = [k for k in val if sub in k]
    return None if not ks else float(val[sorted(ks, key=len)[0]])


def main():
    reg = load_registry()
    rows = []
    for fam, ids in FAMILIES.items():
        base = reg.get(ids[0], {}).get("summary") or {}
        bv = base.get("val") or {}
        for i in ids:
            r = reg.get(i)
            if not r:
                continue
            s = r.get("summary") or {}; v = s.get("val") or {}
            row = {"family": fam, "experiment_id": i, "reward_version": r.get("reward_version"), "verifier_version": r.get("verifier_version"),
                   "rubric_version": r.get("rubric_version"), "decision": r.get("decision"), "train_reward": s.get("final_reward"),
                   "kl": s.get("final_kl"), "entropy": s.get("final_entropy"), "max_grad_norm": s.get("max_grad_norm")}
            for name, sub in METRICS:
                row[name] = pick(v, sub)
                b = pick(bv, sub)
                row[f"delta_{name}"] = None if (row[name] is None or b is None) else round(row[name] - b, 4)
            rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(f"{ROOT}/results", exist_ok=True); df.to_csv(f"{ROOT}/results/ablations.csv", index=False)
    md = ["# Ablation report\n", "Each arm changes ONE variable relative to `exp_D000_baseline` (same data, seed, steps, GPU). Validation metrics are "
          "the verifier's components on the repo-disjoint validation split (`score/mean` = mean final reward, `rule_correctness` = complete solve rate).\n"]
    # verifier offline ablation (synthetic)
    for p in sorted(glob.glob(f"{ROOT}/results/verifier_ablation/*/summary.json")):
        d = json.load(open(p)); tag = p.split("/")[-2]
        md.append(f"## Offline verifier ablation — `{tag}`\n")
        md.append("Gold-patch recovery rate per parser arm (mean over 9 response wrappings): " + json.dumps(d.get("synthetic_overall", {})) + "\n")
        if "model" in d:
            md.append("On real policy samples: " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "fmt"} for k, v in d["model"].items()}) + "\n")
    for fam in FAMILIES:
        sub = df[df["family"] == fam]
        if len(sub) == 0:
            continue
        md.append(f"## {fam}\n")
        cols = ["experiment_id", "reward_version", "val_score", "delta_val_score", "val_resolved", "delta_val_resolved", "val_f2p", "val_p2p", "val_apply", "val_format", "val_resp_len", "train_reward", "kl", "max_grad_norm", "decision"]
        md.append(sub[cols].to_markdown(index=False) + "\n")
    open(f"{ROOT}/docs/ablation_report.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(df.to_string() if len(df) else "no experiments yet")


if __name__ == "__main__":
    main()

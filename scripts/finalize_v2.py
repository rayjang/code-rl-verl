#!/usr/bin/env python3
"""Decide the prompt repair from before/after validation p-hat and build curated_v2 (band-filtered by the
base-policy train p-hat). Prints the decision table; writes docs/dataset_quality_report.md for curated_v2."""
import json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def summ(path):
    return json.load(open(path))["summary"] if os.path.exists(path) else None


def main():
    before = summ(f"{ROOT}/results/phat/curated_v1_validation_k8/summary.json")
    after = summ(f"{ROOT}/results/phat/curated_v2pre_validation_k8/summary.json")
    assert before and after, "need both validation p-hat runs"
    rows = []
    for vt in sorted(set(before) | set(after)):
        b, a = before.get(vt, {}), after.get(vt, {})
        rows.append((vt, b.get("mean_p_hat"), a.get("mean_p_hat"), b.get("frac_band_0.2_0.8"), a.get("frac_band_0.2_0.8"), b.get("mean_reward"), a.get("mean_reward")))
    print("variant       p_hat(v1)  p_hat(v2pre)  band(v1)  band(v2pre)  reward(v1)  reward(v2pre)")
    for r in rows:
        print(f"{r[0]:12s} {r[1]:9.3f} {r[2]:12.3f} {r[3]:9.2f} {r[4]:11.2f} {r[5]:10.3f} {r[6]:13.3f}")
    gain = sum((after.get(vt, {}).get("mean_p_hat", 0) - before.get(vt, {}).get("mean_p_hat", 0)) for vt in ("ut_pytest", "ut_function"))
    use_fix = gain > 0.02
    print(f"prompt-repair decision: {'ADOPT' if use_fix else 'REJECT'} (pytest+function mean p_hat gain = {gain:+.3f})")
    args = [PY, f"{ROOT}/scripts/build_dataset.py", "--version", "curated_v2", "--seed", "0", "--phat", f"{ROOT}/results/phat/curated_v1_train_k8/phat.jsonl",
            "--band", "0.05,0.95", "--hard_rule", "no_signal", "--verifier_version", "vf_v002"]
    if use_fix:
        args.append("--fix_entry_point")
    print(" ".join(args)); subprocess.run(args, check=True)
    subprocess.run([PY, f"{ROOT}/scripts/report_dataset_quality.py", "--version", "curated_v2"], check=True)
    json.dump({"prompt_fix_adopted": use_fix, "gain_pytest_function": gain, "rows": rows}, open(f"{ROOT}/data/curated/curated_v2/prompt_fix_decision.json", "w"), indent=1)


if __name__ == "__main__":
    main()

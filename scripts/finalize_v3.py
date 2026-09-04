#!/usr/bin/env python3
"""Qwen3-30B-A3B: build curated_v3 from the Qwen3 base p-hat (train-only band filter), no 32k context cap.
Prompt repair decision is inherited from curated_v2 (adopted). Original docstring: Decide the prompt repair from before/after validation p-hat and build curated_v2 (band-filtered by the
base-policy train p-hat). Prints the decision table; writes docs/dataset_quality_report.md for curated_v2."""
import json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def summ(path):
    return json.load(open(path))["summary"] if os.path.exists(path) else None


def main():
    before = summ(f"{ROOT}/results/phat/curated_v2_validation_base_k8/summary.json")   # Qwen1.5-MoE, same prompts (32k rows)
    after = summ(f"{ROOT}/results/phat/curated_v3pre_validation_q3_k8/summary.json")   # Qwen3-30B-A3B
    assert after, "need the Qwen3 validation p-hat run"
    before = before or {}
    rows = []
    for vt in sorted(set(before) | set(after)):
        b, a = before.get(vt, {}), after.get(vt, {})
        rows.append((vt, b.get("mean_p_hat"), a.get("mean_p_hat"), b.get("frac_band_0.2_0.8"), a.get("frac_band_0.2_0.8"), b.get("mean_reward"), a.get("mean_reward")))
    print("variant       p_hat(v1)  p_hat(v2pre)  band(v1)  band(v2pre)  reward(v1)  reward(v2pre)")
    for r in rows:
        print(f"{r[0]:12s} {r[1]:9.3f} {r[2]:12.3f} {r[3]:9.2f} {r[4]:11.2f} {r[5]:10.3f} {r[6]:13.3f}")
    gain = sum((after.get(vt, {}).get("mean_p_hat", 0) - before.get(vt, {}).get("mean_p_hat", 0)) for vt in ("ut_pytest", "ut_function"))
    use_fix = True   # inherited from the curated_v2 decision (+0.144 on Qwen1.5-MoE); the table above is model-vs-model
    print(f"model switch effect on validation (Qwen1.5 -> Qwen3), pytest+function mean p_hat gain = {gain:+.3f}")
    args = [PY, f"{ROOT}/scripts/build_dataset.py", "--version", "curated_v3", "--seed", "0", "--phat", f"{ROOT}/results/phat/curated_v3pre_train_q3_k8/phat.jsonl",
            "--band", "0.05,0.95", "--hard_rule", "no_signal", "--verifier_version", "vf_v002", "--max_prompt_tokens", "129024", "--tokenizer", "/scratch/r919a03/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"]
    if use_fix:
        args.append("--fix_entry_point")
    print(" ".join(args)); subprocess.run(args, check=True)
    subprocess.run([PY, f"{ROOT}/scripts/report_dataset_quality.py", "--version", "curated_v3"], check=True)
    json.dump({"prompt_fix_adopted": use_fix, "gain_pytest_function": gain, "rows": rows}, open(f"{ROOT}/data/curated/curated_v3/prompt_fix_decision.json", "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Single-file difficulty-annotated dataset (no train/val/test split).

Rows = every environment-valid instance whose difficulty was MEASURED with the current policy model
(Qwen3-30B-A3B-Instruct-2507, k=8 samples, temperature 1.0, verifier vf_v002, reward rw_v001):
curated_v3pre train+validation+test (6,634 rows) [+ the 16,464-row extended unit-test set if measured, --include_ext].

Adds per-row: empirical_success_rate (p_hat), difficulty (bin), difficulty_score (0..10 = round(10*(1-p_hat))),
reward_mean/std, partial_signal, original_split, train_recommended (band 0.05<=p_hat<=0.95 or partial signal).
Writes data/curated_v4_single/{dataset.parquet, dataset.jsonl.gz, dataset_manifest.json, DATASET.md}.
"""
import argparse, collections, gzip, hashlib, json, os, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BINS = [  # (name, lo, hi] on p_hat
    ("unsolved_no_signal", None, None),  # p_hat == 0 and reward_std == 0
    ("very_hard", 0.0, 0.05),            # p_hat < 0.05 (but some reward signal)
    ("hard", 0.05, 0.2),
    ("medium", 0.2, 0.5),
    ("easy", 0.5, 0.8),
    ("very_easy", 0.8, 0.95),
    ("trivial", 0.95, 1.01),
]


def bin_of(p, std):
    if p == 0 and (std or 0) == 0:
        return "unsolved_no_signal"
    for name, lo, hi in BINS[1:]:
        if lo <= p < hi:
            return name
    return "trivial"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="curated_v4_single")
    ap.add_argument("--src", default="curated_v3pre")
    ap.add_argument("--phat_tag", default="q3_k8")
    ap.add_argument("--include_ext", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    a = ap.parse_args()
    out = f"{ROOT}/data/curated/{a.version}"; os.makedirs(out, exist_ok=True)
    frames = []
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(f"{ROOT}/data/curated/{a.src}/{split}.parquet"); df["_split"] = split
        ph = {r["instance_id"]: r for r in load_jsonl(f"{ROOT}/results/phat/{a.src}_{split}_{a.phat_tag}/phat.jsonl")}
        df["_ph"] = [ph.get(dict(e)["instance_id"]) for e in df["extra_info"]]
        frames.append(df)
    if a.include_ext and os.path.exists(f"{ROOT}/results/phat/ext_{a.phat_tag}/phat.jsonl"):
        df = pd.read_parquet(f"{ROOT}/data/processed/ext_fixed.parquet"); df["_split"] = "extended"
        ph = {r["instance_id"]: r for r in load_jsonl(f"{ROOT}/results/phat/ext_{a.phat_tag}/phat.jsonl")}
        df["_ph"] = [ph.get(dict(e)["instance_id"]) for e in df["extra_info"]]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    n_before = len(df)
    df = df[df["_ph"].notna()].reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        e = dict(r["extra_info"]); p = r["_ph"]
        pv = float(p["p_hat"]); std = float(p.get("reward_std", 0.0) or 0.0)
        e.update({"empirical_success_rate": pv, "empirical_n": int(p.get("n", 0)), "reward_mean": float(p.get("mean_reward", 0.0)),
                  "reward_std": std, "partial_signal": bool(std > 0), "difficulty": bin_of(pv, std), "difficulty_score": int(round(10 * (1 - pv))),
                  "difficulty_model": a.model, "difficulty_protocol": f"k={p.get('n', 8)} samples, temperature 1.0, verifier vf_v002, reward rw_v001",
                  "max_f2p_frac": float(p.get("max_f2p_frac", 0.0)), "apply_rate": float(p.get("apply_rate", 0.0)),
                  "original_split": r["_split"], "train_recommended": bool((0.05 <= pv <= 0.95) or (pv < 0.05 and std > 0)),
                  "dataset_version": a.version, "split": "all"})
        e.pop("data_quality_status", None); e.pop("exclusion_reason", None)
        e["task_type"] = e.get("task_type") or ("swe_patch" if str(e.get("variant_type", "")).startswith("swe") else f"unittest_{e.get('harness', '')}")
        rows.append(e)
    df["extra_info"] = rows
    cols = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    df[cols].to_parquet(f"{out}/dataset.parquet", index=False)
    with gzip.open(f"{out}/dataset.jsonl.gz", "wt", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({"data_source": r["data_source"], "prompt": [dict(m) for m in r["prompt"]], "ability": r["ability"],
                                "reward_model": dict(r["reward_model"]), "extra_info": r["extra_info"]}, ensure_ascii=False, default=str) + "\n")
    ei = pd.DataFrame(rows)
    man = {"version": a.version, "rows": int(len(df)), "rows_dropped_no_difficulty": int(n_before - len(df)), "source_splits": dict(collections.Counter(ei["original_split"])),
           "difficulty_model": a.model, "by_task": {}, "by_difficulty": {}, "by_task_difficulty": {}, "train_recommended": int(ei["train_recommended"].sum())}
    for t, g in ei.groupby("task_type"):
        man["by_task"][t] = {"n": int(len(g)), "mean_p_hat": round(float(g["empirical_success_rate"].mean()), 4), "train_recommended": int(g["train_recommended"].sum())}
    for d, g in ei.groupby("difficulty"):
        man["by_difficulty"][d] = int(len(g))
    for (t, d), g in ei.groupby(["task_type", "difficulty"]):
        man["by_task_difficulty"][f"{t}|{d}"] = int(len(g))
    man["sha256"] = {fn: hashlib.sha256(open(f"{out}/{fn}", "rb").read()).hexdigest()[:16] for fn in ("dataset.parquet", "dataset.jsonl.gz")}
    json.dump(man, open(f"{out}/dataset_manifest.json", "w"), indent=1, ensure_ascii=False)
    ei[["instance_id", "task_type", "variant_type", "lang", "repo", "original_split", "empirical_success_rate", "reward_mean", "reward_std", "difficulty", "difficulty_score",
        "train_recommended", "n_f2p_effective", "n_p2p_effective", "prompt_tokens"]].to_csv(f"{out}/instance_difficulty.csv", index=False)
    print(json.dumps({k: man[k] for k in ("rows", "rows_dropped_no_difficulty", "source_splits", "by_task", "by_difficulty", "train_recommended")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

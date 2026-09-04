#!/usr/bin/env python3
"""Prepare the 16,464-row extended unit-test set (unmeasured difficulty) for base-policy measurement:
apply the same entry-point prompt repair and write data/processed/ext_fixed.parquet (+ index dir)."""
import json, os, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scripts.build_dataset import required_names, load_jsonl  # noqa: E402
SRC = f"{ROOT}/sources/rl_code_v1/data"
df = pd.read_parquet(f"{SRC}/parquet/t15_code_rl_extended_train.parquet")
full = {r["instance_id"]: r for r in load_jsonl(f"{SRC}/index/t15_code_unittest_ext.full.jsonl")}
fixed, prompts = 0, []
for _, row in df.iterrows():
    msgs = [dict(m) for m in row["prompt"]]; ei = dict(row["extra_info"]); inst = full.get(ei["instance_id"])
    names = required_names(inst) if inst else []
    if names:
        ko = ei.get("lang") == "ko"
        line = ("\n\n테스트가 사용하는 이름을 정확히 이 이름으로 정의하세요: " if ko else "\n\nThe hidden tests use exactly these names; define them: ") + ", ".join(f"`{n}`" for n in names)
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs[i]["content"] = msgs[i]["content"].rstrip() + line; break
        fixed += 1
    prompts.append(msgs)
df["prompt"] = prompts
os.makedirs(f"{ROOT}/data/processed", exist_ok=True)
df.to_parquet(f"{ROOT}/data/processed/ext_fixed.parquet", index=False)
print("rows", len(df), "prompt fix applied", fixed)

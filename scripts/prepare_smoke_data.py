"""Build a tiny verl dataset for the RL smoke test: N unit-test + M swe_32k rows from core train,
copied verbatim from rl_code_v1 parquet (prompt/reward_model/extra_info untouched)."""
import argparse, os, sys, json
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{ROOT}/sources/rl_code_v1/data/parquet"
ap = argparse.ArgumentParser(); ap.add_argument("--n_ut", type=int, default=48); ap.add_argument("--n_swe", type=int, default=16)
ap.add_argument("--out", default=f"{ROOT}/data/processed/smoke"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
df = pd.read_parquet(f"{SRC}/t15_code_rl_core_train.parquet")
ei = pd.DataFrame(list(df["extra_info"]))
ut = df[ei["variant_type"].str.startswith("ut").values].sample(n=a.n_ut, random_state=a.seed)
swe = df[(ei["variant_type"] == "swe_32k").values].sample(n=a.n_swe, random_state=a.seed)
train = pd.concat([ut, swe]).sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
val = pd.concat([df[ei["variant_type"].str.startswith("ut").values].sample(n=16, random_state=a.seed + 1),
                 df[(ei["variant_type"] == "swe_32k").values].sample(n=4, random_state=a.seed + 1)]).reset_index(drop=True)
train.to_parquet(f"{a.out}/train.parquet", index=False); val.to_parquet(f"{a.out}/val.parquet", index=False)
print("train", len(train), dict(pd.DataFrame(list(train["extra_info"]))["variant_type"].value_counts()), "val", len(val))
print("columns", list(train.columns))

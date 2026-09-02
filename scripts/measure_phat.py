#!/usr/bin/env python3
"""Empirical difficulty (pass@k / success probability) with OUR policy model + verifier, offline with vLLM.

Usage (inside a SLURM GPU job):
  python scripts/measure_phat.py --parquet data/curated/curated_v1/train.parquet --k 8 --out results/phat/train_k8 \
      [--lora <adapter_dir>] [--limit N] [--gpus 6] [--reward configs/reward/rw_v001_baseline.yaml] [--verifier configs/verifier/vf_v002.yaml]
Runs one vLLM engine per GPU (data parallel via subprocesses), scores every sample with verifier.verl_reward
(same code path as training) and writes:
  <out>/responses.jsonl   one record per sample (instance_id, sample_idx, response, reward dict)
  <out>/phat.jsonl        one record per instance (p_hat = mean resolved, mean reward, format stats)
  <out>/summary.json
"""
from __future__ import annotations

import argparse, collections, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
S_MODEL = "/scratch/r919a03/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"


def worker(a):
    """Runs on ONE GPU: generate k samples for its shard, then score them."""
    import pandas as pd
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    df = pd.read_parquet(a.parquet)
    if a.limit:
        df = df.iloc[: a.limit]
    df = df.iloc[a.shard::a.nshards]
    tok_kw = {}
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.85, enforce_eager=True,
              enable_lora=bool(a.lora), max_lora_rank=64, seed=a.seed + a.shard, max_num_seqs=a.max_num_seqs)
    tok = llm.get_tokenizer()
    prompts, meta = [], []
    for _, row in df.iterrows():
        msgs = [dict(m) for m in row["prompt"]]
        ei = dict(row["extra_info"])
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n_tok = len(tok(text).input_ids)
        if n_tok + a.max_tokens > a.max_model_len:
            continue
        prompts.append(text); meta.append((row["data_source"], ei, row["reward_model"]["ground_truth"], n_tok))
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed + a.shard)
    t0 = time.time()
    lora_req = LoRARequest("adapter", 1, a.lora) if a.lora else None
    outs = llm.generate(prompts, sp, lora_request=lora_req)
    gen_s = time.time() - t0
    print(f"[shard {a.shard}] generated {len(outs)}x{a.k} in {gen_s:.0f}s", flush=True)
    del llm
    # ---- score with the verifier (thread pool) ----
    os.environ.setdefault("VERIFIER_CONCURRENCY", str(a.concurrency))
    from concurrent.futures import ThreadPoolExecutor
    from verifier.verl_reward import compute_score_dict
    jobs = []
    for (ds, ei, gt, n_tok), o in zip(meta, outs):
        for j, c in enumerate(o.outputs):
            jobs.append((ds, ei, gt, n_tok, j, c.text, len(c.token_ids), c.finish_reason))
    def score(job):
        ds, ei, gt, n_tok, j, text, n_out, fin = job
        ei2 = {**ei, "valid_response_length": n_out, "max_response_length": a.max_tokens}
        r = compute_score_dict(ds, text, gt, ei2)
        return {"instance_id": ei.get("instance_id"), "data_source": ds, "variant_type": ei.get("variant_type"), "sample_idx": j,
                "prompt_tokens": n_tok, "response_tokens": n_out, "finish_reason": fin, "response": text if a.keep_text else text[:4000], **r}
    t1 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        recs = list(ex.map(score, jobs))
    print(f"[shard {a.shard}] scored {len(recs)} in {time.time() - t1:.0f}s", flush=True)
    with open(f"{a.out}/responses.shard{a.shard}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def aggregate(a):
    recs = []
    for fn in sorted(os.listdir(a.out)):
        if fn.startswith("responses.shard"):
            recs += [json.loads(l) for l in open(os.path.join(a.out, fn), encoding="utf-8")]
    with open(f"{a.out}/responses.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = collections.defaultdict(list)
    for r in recs:
        by[r["instance_id"]].append(r)
    phat, summ = [], collections.defaultdict(list)
    for iid, rs in by.items():
        n = len(rs); ok = [r for r in rs if not r.get("infra_excluded")]
        rw = [r["final_reward"] for r in ok]
        mu = sum(rw) / max(1, len(rw))
        rec = {"instance_id": iid, "variant_type": rs[0]["variant_type"], "n": len(ok), "n_infra": n - len(ok),
               "p_hat": sum(r["rule_correctness_score"] for r in ok) / max(1, len(ok)),
               "mean_reward": mu, "reward_std": (sum((x - mu) ** 2 for x in rw) / max(1, len(rw))) ** 0.5,
               "frac_partial": sum(1 for r in ok if 0 < r["f2p_frac"] < 1) / max(1, len(ok)),
               "max_f2p_frac": max([r["f2p_frac"] for r in ok] or [0.0]),
               "mean_f2p_frac": sum(r["f2p_frac"] for r in ok) / max(1, len(ok)),
               "extract_rate": sum(r["patch_extraction_score"] for r in ok) / max(1, len(ok)),
               "format_rate": sum(r["patch_format_score"] for r in ok) / max(1, len(ok)),
               "apply_rate": sum(r["patch_apply_score"] for r in ok) / max(1, len(ok)),
               "mean_resp_tokens": sum(r["response_tokens"] for r in rs) / n,
               "err_kinds": dict(collections.Counter(r["err_kind"] for r in rs))}
        phat.append(rec); summ[rec["variant_type"]].append(rec)
    with open(f"{a.out}/phat.jsonl", "w", encoding="utf-8") as f:
        for r in phat:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {}
    for vt, rs in summ.items():
        ps = [r["p_hat"] for r in rs]
        summary[vt] = {"n_instances": len(rs), "mean_p_hat": sum(ps) / len(ps), "frac_zero": sum(p == 0 for p in ps) / len(ps),
                       "frac_zero_and_no_signal": sum(1 for r in rs if r["p_hat"] == 0 and r["reward_std"] == 0) / len(rs),
                       "frac_reward_std_zero": sum(1 for r in rs if r["reward_std"] == 0) / len(rs),
                       "frac_one": sum(p == 1 for p in ps) / len(ps), "frac_band_0.2_0.8": sum(0.2 <= p <= 0.8 for p in ps) / len(ps),
                       "mean_reward": sum(r["mean_reward"] for r in rs) / len(rs), "extract_rate": sum(r["extract_rate"] for r in rs) / len(rs),
                       "apply_rate": sum(r["apply_rate"] for r in rs) / len(rs), "mean_resp_tokens": sum(r["mean_resp_tokens"] for r in rs) / len(rs)}
    json.dump({"args": vars(a), "summary": summary, "n_responses": len(recs)}, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=int(os.environ.get("NGPU", "6")))
    ap.add_argument("--model", default=S_MODEL); ap.add_argument("--lora", default="")
    ap.add_argument("--max_model_len", type=int, default=32768); ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--keep_text", action="store_true")
    ap.add_argument("--shard", type=int, default=-1); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--reward", default=""); ap.add_argument("--verifier", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.reward:
        os.environ["REWARD_CONFIG"] = a.reward
    if a.verifier:
        os.environ["VERIFIER_CONFIG"] = a.verifier
    if a.shard >= 0:
        worker(a); return
    # launch one worker per GPU
    procs = []
    for g in range(a.gpus):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
        cmd = [sys.executable, __file__] + [x for x in sys.argv[1:]] + ["--shard", str(g), "--nshards", str(a.gpus)]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(f"{a.out}/worker{g}.log", "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    print("worker rcs", rcs)
    aggregate(a)


if __name__ == "__main__":
    main()

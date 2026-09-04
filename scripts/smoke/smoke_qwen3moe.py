"""Qwen3-30B-A3B smoke on ONE GPU: module inventory, LoRA candidates (params, memory), 32k+ prompt forward,
and vLLM generation (eager) with a 100k-token prompt. Writes results/smoke/smoke_qwen3moe.json"""
import json, os, sys, time, gc, re, collections, torch
S = os.environ["MODEL_PATH"]; OUT = "/scratch/r919a03/code_verl_test/results/smoke/smoke_qwen3moe.json"


def main():
    import transformers, peft, vllm, verl
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights
    res = {"model": S, "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "vllm": vllm.__version__, "verl": verl.__version__}, "gpu": torch.cuda.get_device_name(0)}
    cfg = AutoConfig.from_pretrained(S)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    kinds = collections.Counter(); pcount = {}
    for n, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear):
            k = re.sub(r"\.\d+\.", ".N.", n); kinds[k] += 1; pcount[k] = mod.in_features * mod.out_features
    res["linear_modules"] = {k: {"count": v, "params_each": pcount[k]} for k, v in sorted(kinds.items())}
    res["expert_params_layer0"] = [n for n, _ in m.named_parameters() if ".layers.0." in n and "experts" in n]
    res["total_params"] = sum(p.numel() for p in m.parameters())
    print(json.dumps({k: res[k] for k in ("linear_modules", "expert_params_layer0", "total_params")}, indent=1)[:2000], flush=True)
    del m; gc.collect()
    # ---- vLLM long prompt ----
    import subprocess
    p = subprocess.run([sys.executable, "-c", f"""
import os, time, json
from vllm import LLM, SamplingParams
llm = LLM(model={S!r}, dtype='bfloat16', max_model_len=140000, gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=64, enforce_eager=True, enable_prefix_caching=True)
tok = llm.get_tokenizer()
import pandas as pd
df = pd.read_parquet('/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/parquet/t15_code_rl_core_train.parquet')
ei = [dict(x) for x in df['extra_info']]
rows = [i for i,e in enumerate(ei) if e['variant_type']=='swe_128k'][:2] + [i for i,e in enumerate(ei) if e['variant_type']=='swe_32k'][:2]
prompts = [tok.apply_chat_template([dict(m) for m in df.iloc[i]['prompt']], tokenize=False, add_generation_prompt=True) for i in rows]
lens = [len(tok(p).input_ids) for p in prompts]
t0=time.time(); outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.8, max_tokens=1024, seed=0)); dt=time.time()-t0
print('VLLM_RESULT ' + json.dumps({{'prompt_tokens': lens, 'gen_s': round(dt,1), 'out_tokens': [len(o.outputs[0].token_ids) for o in outs], 'samples': [o.outputs[0].text[:600] for o in outs]}}, ensure_ascii=False))
"""], capture_output=True, text=True, timeout=3000)
    line = next((l for l in p.stdout.splitlines() if l.startswith("VLLM_RESULT ")), None)
    res["vllm"] = json.loads(line[len("VLLM_RESULT "):]) if line else {"error": (p.stdout + p.stderr)[-2500:]}
    print("VLLM", json.dumps(res["vllm"], ensure_ascii=False)[:1500], flush=True)
    # ---- HF + LoRA memory on long sequence ----
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(S)
    res["lora"] = {}
    for name, kw in {"A_attn": dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])}.items():
        torch.cuda.reset_peak_memory_stats(); gc.collect(); torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(S, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
        model.gradient_checkpointing_enable()
        pm = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, task_type="CAUSAL_LM", **kw))
        trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad); total = sum(p.numel() for p in pm.parameters())
        out = {"trainable": trainable, "total": total, "trainable_pct": round(100 * trainable / total, 3)}
        for L in (8192, 32768):
            try:
                ids = torch.randint(100, 30000, (1, L), device="cuda")
                torch.cuda.reset_peak_memory_stats(); pm.train(); o = pm(input_ids=ids, labels=ids); o.loss.backward(); pm.zero_grad(set_to_none=True)
                out[f"peak_gib_seq{L}"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            except Exception as e:
                out[f"peak_gib_seq{L}"] = f"error: {type(e).__name__}: {str(e)[:120]}"
            gc.collect(); torch.cuda.empty_cache()
        res["lora"][name] = out; print("LORA", name, json.dumps(out), flush=True)
        del pm, model; gc.collect(); torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False); print("SMOKE_DONE", OUT)


if __name__ == "__main__":
    main()

"""Smoke test 1 (single GPU): (a) vLLM generation on Qwen1.5-MoE-A2.7B-Chat, (b) HF+PEFT LoRA wrap for
candidate target sets, printing trainable/total params and peak memory for a short fwd/bwd.
Writes JSON to results/smoke/smoke_infer_lora.json"""
import json, os, sys, time, torch, gc
S = os.environ.get("MODEL_PATH", "/scratch/r919a03/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be")
OUT = "/scratch/r919a03/code_verl_test/results/smoke/smoke_infer_lora.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
def main():
    res = {"host": os.uname().nodename, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "n_gpu": torch.cuda.device_count()}
    import transformers, peft, vllm, verl
    res["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "peft": peft.__version__, "vllm": vllm.__version__, "verl": verl.__version__}
    res["gpu"] = torch.cuda.get_device_name(0)

    # ---------- (a) vLLM generation ----------
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=S, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.5, enable_lora=True, max_lora_rank=64, enforce_eager=False)
    tok = llm.get_tokenizer()
    msgs = [[{"role": "user", "content": "Write a Python function that returns the longest common prefix of a list of strings. Reply with a single ```python code block."}],
            [{"role": "user", "content": "1+1=?"}]]
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256, seed=0))
    res["vllm"] = {"load_s": round(time.time() - t0, 1), "outputs": [o.outputs[0].text[:400] for o in outs], "n_tokens": [len(o.outputs[0].token_ids) for o in outs]}
    print("VLLM OK", json.dumps(res["vllm"], ensure_ascii=False)[:600], flush=True)
    del llm; gc.collect(); torch.cuda.empty_cache()
    try:
        import ray; ray.shutdown()
    except Exception: pass

    # ---------- (b) HF + PEFT LoRA candidates ----------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    hf_tok = AutoTokenizer.from_pretrained(S)
    CANDS = {
        "A_attn": dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
        "B_attn_shared": dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj"]),
        "C_all_linear_minus_gates": dict(target_modules="all-linear", exclude_modules=["shared_expert_gate", "lm_head"]),
    }
    res["lora"] = {}
    for name, kw in CANDS.items():
        torch.cuda.reset_peak_memory_stats(); gc.collect(); torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(S, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        base_total = sum(p.numel() for p in model.parameters())
        cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", **kw)
        try:
            pm = get_peft_model(model, cfg)
            trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
            total = sum(p.numel() for p in pm.parameters())
            lora_mods = sorted({n.split(".")[-1] if "lora" not in n else n for n, m in pm.named_modules() if "lora_A" in n and n.endswith("lora_A")})
            wrapped = sorted({n.replace("base_model.model.", "").split(".lora_A")[0].replace("model.layers.0.", "L0.") for n, _ in pm.named_modules() if n.endswith("lora_A") and ".layers.0." in n})
            enc = hf_tok(prompts[0] * 8, return_tensors="pt").to("cuda")
            pm.train(); out = pm(**enc, labels=enc["input_ids"]); out.loss.backward()
            peak = torch.cuda.max_memory_allocated() / 2**30
            res["lora"][name] = {"trainable": trainable, "total": total, "base_total": base_total, "trainable_pct": round(100 * trainable / total, 3), "peak_mem_gib_seq%d" % enc["input_ids"].shape[1]: round(peak, 1), "layer0_wrapped": wrapped, "loss": float(out.loss)}
            print("LORA", name, json.dumps(res["lora"][name])[:800], flush=True)
            del pm, out
        except Exception as e:
            res["lora"][name] = {"error": repr(e)[:500]}; print("LORA", name, "ERROR", repr(e)[:300], flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    # (c) experts via target_parameters (peft >= 0.17) — informational only
    try:
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(S, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        pnames = [n for n, p in model.named_parameters() if ".experts." in n and ".layers.0." in n]
        cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], target_parameters=[n.split("model.layers.0.")[1] for n in pnames] if pnames else None, task_type="CAUSAL_LM")
        pm = get_peft_model(model, cfg)
        trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
        res["lora"]["D_experts_target_parameters"] = {"expert_param_names_layer0": pnames, "trainable": trainable, "note": "peft target_parameters on fused expert tensors; vLLM sync path unverified"}
        print("LORA D", json.dumps(res["lora"]["D_experts_target_parameters"])[:600], flush=True)
        del pm, model
    except Exception as e:
        res["lora"]["D_experts_target_parameters"] = {"error": repr(e)[:500]}; print("LORA D ERROR", repr(e)[:300])
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("SMOKE_DONE", OUT)


if __name__ == '__main__':
    main()

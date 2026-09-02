"""vLLM generation smoke for one engine configuration (run in its own process)."""
import json, os, sys, time
S = "/scratch/r919a03/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"


def main():
    variant = sys.argv[1]
    out_path = sys.argv[2]
    kw = dict(model=S, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.5, enable_lora=True, max_lora_rank=64)
    if variant == "eager":
        kw["enforce_eager"] = True
    elif variant == "compile_nopad":
        os.environ["TORCHINDUCTOR_COMPREHENSIVE_PADDING"] = "0"
    elif variant == "nolora":
        kw.pop("enable_lora"); kw.pop("max_lora_rank")
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    msgs = [[{"role": "user", "content": "Write a Python function that returns the longest common prefix of a list of strings. Reply with a single ```python code block."}],
            [{"role": "user", "content": "What is 17*23? Answer briefly."}]]
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    t1 = time.time()
    outs = llm.generate(prompts * 16, SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256, seed=0))
    gen_s = time.time() - t1
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    res = {"variant": variant, "ok": True, "load_s": round(t1 - t0, 1), "gen_s": round(gen_s, 1), "tok_per_s": round(ntok / gen_s, 1),
           "samples": [o.outputs[0].text[:300] for o in outs[:2]]}
    json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
    print("VARIANT_OK", json.dumps(res, ensure_ascii=False)[:500])


if __name__ == "__main__":
    main()

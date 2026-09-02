import importlib.util, sys, json
for label, path in (("bundle","/scratch/r919a03/code_verl_test/sources/t01_math_reward_rubric/t01_math_reward.py"),
                    ("rl_code_v1","/scratch/r919a03/code_verl_test/sources/rl_code_v1/reward/math/t01_math_reward.py")):
    spec=importlib.util.spec_from_file_location("m_"+label, path); m=importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:
        print(label,"IMPORT FAIL",type(e).__name__,e); continue
    d=m.compute_score_dict("t01_math","\\boxed{1}","1",{"valid_response_length":1,"max_response_length":15360})
    print(label,"n_keys",len(d)); print(label,"keys in return order:",list(d.keys()))
    print(label,"value types:",sorted(set(type(v).__name__ for v in d.values())))
    if hasattr(m,"KEYS"): print(label,"KEYS sorted:",m.KEYS, len(m.KEYS))
    if hasattr(m,"_KEYS"): print(label,"_KEYS:",m._KEYS,len(m._KEYS), "matches returned set?", set(m._KEYS)==set(d))
    if hasattr(m,"TIER_NAMES"): print(label,"TIER_NAMES",m.TIER_NAMES, "TIER_NOMATCH",m.TIER_NOMATCH)
    if hasattr(m,"extractor_names"): print(label,"extractors",m.extractor_names(),"TIER_NONE",m.TIER_NONE)
    if hasattr(m,"step_names"): print(label,"norm steps",m.step_names(), len(m.step_names()))
    if hasattr(m,"symbolic_backend"): print(label,"symbolic backend",m.symbolic_backend())
    # invariants demo
    for resp,gt in (("\\boxed{\\frac{1}{2}}","0.5"),("Answer: 42","42"),("no answer here","7"),("\\boxed{x+1}","1+x"),("\\boxed{3} or \\boxed{4}","4")):
        r=m.compute_score_dict("t01_math",resp,gt,{"valid_response_length":10,"max_response_length":100})
        print("  ",label,repr(resp),repr(gt),"score",r["score"],"answer_match",r["answer_match"], {k:v for k,v in r.items() if k.startswith("math_") and k in ("math_match_tier","math_extract_tier","math_equiv_path")})
    # determinism
    a=m.compute_score_dict("t01_math","\\boxed{1}","1",{}); b=m.compute_score_dict("t01_math","\\boxed{1}","1",{})
    print(label,"deterministic on repeat:",a==b, "empty extra_info ok: resp_limit",a["resp_limit"])

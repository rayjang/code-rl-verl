import importlib.util, sys, hashlib, json
B="/scratch/r919a03/code_verl_test/sources/t01_math_reward_rubric/t01_math_reward.py"
V="/scratch/r919a03/code_verl_test/sources/rl_code_v1/reward/math/t01_math_reward.py"
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
print("python", sys.version.split()[0])
try:
    import sympy; print("sympy", sympy.__version__)
except Exception as e: print("sympy missing", e)
for mod in ("math_verify","antlr4"):
    try: __import__(mod); print(mod,"AVAILABLE")
    except Exception as e: print(mod,"missing:",type(e).__name__)
b=load("bmod",B); v=load("vmod",V)
ei={"valid_response_length":10,"max_response_length":100}
d=b.compute_score_dict("t01_math","\\boxed{1}","1",ei)
print("B nkeys",len(d)); print("B keys(order)",list(d)); print("B types",set(type(x).__name__ for x in d.values()))
print("B KEYS len",len(b.KEYS))
print("B extractors",b.extractor_names(),"TIER_NONE",b.TIER_NONE)
print("B tiers",b.tier_names(),"TIER_NOMATCH",b.TIER_NOMATCH)
print("B backend",b.symbolic_backend())
dv=v.compute_score_dict("t01_math","\\boxed{1}","1",ei)
print("V nkeys",len(dv)); print("V keys(order)",list(dv)); print("V types",{k:type(x).__name__ for k,x in dv.items() if not isinstance(x,float)})
print("V _KEYS == returned keys?", tuple(dv)==v._KEYS, len(v._KEYS))
# behaviors
for resp,gold in [("\\boxed{3} or \\boxed{4}","4"),("\\boxed{x+1}","1+x"),("\\boxed{3} or \\boxed{4}","3"),("\\boxed{1=2=3=4}","4"),("\\boxed{\\frac{1}{2}}","0.5"),("\\boxed{[-1,1]}","[-1,1)")]:
    rb=b.compute_score_dict("t01_math",resp,gold,ei); rv=v.compute_score_dict("t01_math",resp,gold,ei)
    print(repr(resp),"vs",repr(gold),"B score",rb["score"],"match_tier",rb["math_match_tier"],"sym_used",rb["math_sym_used"],"sym_err",rb["math_sym_error"],"scorer_err",rb["math_scorer_error"],"| V score",rv["score"],rv["math_equiv_path"])
# determinism
r1=b.compute_score_dict("t01_math","\\boxed{x+1}","1+x",ei); r2=b.compute_score_dict("t01_math","\\boxed{x+1}","1+x",ei); print("B deterministic",r1==r2)
# gt_kind codes
print("gt_kind codes", b.GT_INT,b.GT_NUM,b.GT_TUPLE,b.GT_SET,b.GT_REL,b.GT_EXPR,b.GT_OTHER)
# ground_truth list
print("B list gt", b.compute_score_dict("t01_math","\\boxed{7}",["7","8"],ei)["answer_match"])
# no length info, buffer off
print("B no ei", b.compute_score_dict("t01_math","\\boxed{7}","7",None)["resp_tokens"], b.compute_score_dict("t01_math","\\boxed{7}","7",None)["resp_limit"])
# parse_latex actually throws?
try:
    from sympy.parsing.latex import parse_latex
    try:
        print("parse_latex ok:", parse_latex(r"\frac{1}{2}"))
    except Exception as e:
        print("parse_latex call throws:",type(e).__name__, str(e)[:100])
except Exception as e:
    print("parse_latex import fails:",type(e).__name__)
print("md5 B", hashlib.md5(open(B,'rb').read()).hexdigest()[:8], "md5 V", hashlib.md5(open(V,'rb').read()).hexdigest()[:8])

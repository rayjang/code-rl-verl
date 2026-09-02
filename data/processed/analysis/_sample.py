import json, glob, os
D='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/index'
def trunc(v,n=300):
    s=json.dumps(v,ensure_ascii=False) if not isinstance(v,str) else v
    return (s[:n]+f'...[len={len(s)}]') if len(s)>n else s
for f in sorted(glob.glob(D+'/*.full.jsonl')):
    r=json.loads(open(f).readline())
    print('=====',os.path.basename(f))
    for k,v in r.items():
        print(f'  {k}: {trunc(v)}')

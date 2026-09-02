import json, sys, collections, glob, os
D='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/index'
for f in sorted(glob.glob(D+'/*.jsonl')):
    rows=[json.loads(l) for l in open(f)]
    keys=collections.OrderedDict()
    for r in rows:
        for k,v in r.items():
            e=keys.setdefault(k,{'types':collections.Counter(),'nonnull':0})
            e['types'][type(v).__name__]+=1
            if v is not None and v!='' and v!=[] and v!={}: e['nonnull']+=1
    print('=====',os.path.basename(f),'rows=',len(rows))
    for k,e in keys.items():
        print(f'  {k:28s} types={dict(e["types"])} nonnull%={100*e["nonnull"]/len(rows):.1f}')

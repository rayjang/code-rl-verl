import json, re, collections
D='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/index'
out={}
for t in ('t15_code_unittest','t15_code_unittest_ext'):
    C=collections.Counter(); ex={}
    for l in open(f'{D}/{t}.full.jsonl',encoding='utf-8'):
        r=json.loads(l); h=r['harness']; st=r['statement']
        if h=='function':
            names=set()
            for x in r['tests']:
                a=x.get('assertion') or ''
                m=re.search(r'assert\s+(?:\w+\s*=\s*)?([A-Za-z_]\w*)\s*[\(\.]',a) or re.search(r'([A-Za-z_]\w*)\s*\(',a)
                if m: names.add(m.group(1))
            names-= {'assert','isinstance','len','str','int','float','list','dict','set','tuple','abs','round','sorted','print','math','type','all','any','bool','sum','max','min','range','map','filter','repr','hasattr','callable','zip','enumerate','next','iter','id'}
            if not names: C['function|no_name_extracted']+=1; continue
            ep=r['entry_point']
            hit=[n for n in names if n in st]
            if ep and ep in st: C['function|entry_point_in_statement']+=1
            elif hit: C['function|called_name_in_statement']+=1
            else: C['function|called_name_NOT_in_statement']+=1; ex.setdefault('function_missing',(r['instance_id'],sorted(names)[:3],st[:150]))
        elif h=='pytest':
            imp=set()
            for x in r['tests']:
                a=x.get('assertion') or ''
                for m in re.finditer(r'^\s*from\s+solution\s+import\s+([^\n]+)',a,re.M):
                    for n in re.split(r'[,\s()\\]+',m.group(1)):
                        if n and n!='as' and re.match(r'^[A-Za-z_]\w*$',n): imp.add(n)
                if re.search(r'^\s*import\s+solution',a,re.M):
                    imp|=set(re.findall(r'solution\.([A-Za-z_]\w*)',a))
            if not imp: C['pytest|no_import_names']+=1; ex.setdefault('pytest_noimport',(r['instance_id'],(r['tests'][0].get('assertion') or '')[:200])); continue
            ep=r['entry_point']
            if ep and ep in st: C['pytest|entry_point_in_statement']+=1
            elif all(n in st for n in imp): C['pytest|all_imported_names_in_statement']+=1
            elif any(n in st for n in imp): C['pytest|some_imported_names_in_statement']+=1
            else: C['pytest|imported_names_NOT_in_statement']+=1; ex.setdefault('pytest_missing',(r['instance_id'],sorted(imp)[:3],st[:150]))
            C['pytest|entry_point_present']+= bool(ep)
            if ep and ep not in imp: C['pytest|entry_point_not_among_imports']+=1; ex.setdefault('ep_not_imported',(r['instance_id'],ep,sorted(imp)[:3]))
        else:
            C['stdio']+=1
            C['stdio|tests_have_stdin_stdout']+= all('stdin' in x and 'stdout' in x for x in r['tests'])
    out[t]={'counts':dict(C),'examples':ex}
print(json.dumps(out,indent=1,ensure_ascii=False))

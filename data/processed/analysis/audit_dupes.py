#!/usr/bin/env python3
import json, glob, os, re, collections, difflib, itertools, time
D='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/index'
OUT='/scratch/r919a03/code_verl_test/data/processed/analysis'
def load(p): return [json.loads(l) for l in open(p,encoding='utf-8')]
full={t:load(f'{D}/{t}.full.jsonl') for t in ['t15_code_swesmith','t15_code_unittest','t15_code_unittest_ext']}
S6={}
t0=time.time()
F=full['t15_code_swesmith']
byrepo=collections.defaultdict(list)
for r in F: byrepo[r['repo']].append(r)
def shingles(s,k=5):
    w=re.findall(r'\w+',s.lower()); return {' '.join(w[i:i+k]) for i in range(max(0,len(w)-k+1))}
pairs_qr=0; pairs_ratio=0; pairs_jac=0; pairs_jac5=0; top=[]; npairs=0; nratio=0
for repo,rs in byrepo.items():
    sh={r['instance_id']:shingles(r['problem_statement']) for r in rs}
    for a,b in itertools.combinations(rs,2):
        npairs+=1
        sa,sb=sh[a['instance_id']],sh[b['instance_id']]
        jac=len(sa&sb)/len(sa|sb) if (sa|sb) else 0.0
        if jac>0.5: pairs_jac5+=1
        if jac>0.9: pairs_jac+=1
        sm=difflib.SequenceMatcher(None,a['problem_statement'],b['problem_statement'],autojunk=False)
        qr=sm.quick_ratio()
        if qr>0.9:
            pairs_qr+=1
            if jac>0.15 or nratio<3000:
                rr=sm.ratio(); nratio+=1
            else:
                rr=None
            if rr is not None and rr>0.9: pairs_ratio+=1
            top.append((rr if rr is None else round(rr,3),round(qr,3),round(jac,3),a['instance_id'],b['instance_id']))
top.sort(key=lambda x:(x[0] if x[0] is not None else -1, x[2]),reverse=True)
S6['swe']={'method':'group by repo; 5-word shingle Jaccard on problem_statement for every within-repo pair; difflib.SequenceMatcher(autojunk=False).quick_ratio() for every pair; real ratio() only for pairs with quick_ratio>0.9 (all if jaccard>0.15, else first 3000)',
    'n_within_repo_pairs':npairs,'pairs_quick_ratio_gt_0.9':pairs_qr,'n_real_ratio_computed':nratio,'pairs_ratio_gt_0.9':pairs_ratio,'pairs_shingle_jaccard_gt_0.9':pairs_jac,'pairs_shingle_jaccard_gt_0.5':pairs_jac5,
    'top_pairs_(ratio,quick_ratio,jaccard,id_a,id_b)':top[:25],
    'exact_dup_problem_statement_groups':sum(1 for v in collections.Counter(r['problem_statement'] for r in F).values() if v>1),
    'exact_dup_gold_patch_groups':sum(1 for v in collections.Counter(r['gold_patch'] for r in F).values() if v>1),
    'exact_dup_f2p_set_groups':sum(1 for v in collections.Counter(json.dumps(sorted(r['FAIL_TO_PASS'])) for r in F).values() if v>1),
    'exact_dup_f2p_set_rows':sum(v for v in collections.Counter(json.dumps(sorted(r['FAIL_TO_PASS'])) for r in F).values() if v>1),
    'dup_f2p_set_examples':[(k[:300],v) for k,v in collections.Counter(json.dumps(sorted(r['FAIL_TO_PASS'])) for r in F).items() if v>1][:5],
    'seconds':round(time.time()-t0,1)}
print('swe dupes done',S6['swe']['seconds'],flush=True)
def norm(s): return re.sub(r'\s+',' ',(s or '').strip())
ut={}; keys={}
for t in ('t15_code_unittest','t15_code_unittest_ext'):
    Fx=full[t]
    ks=[(r['instance_id'],norm(r['statement']),norm(r['reference_solution']),json.dumps(r['tests'],sort_keys=True)) for r in Fx]
    keys[t]=ks
    st=collections.defaultdict(list); rs=collections.defaultdict(list); ts=collections.defaultdict(list)
    for iid,s,ref,tt in ks:
        st[s].append(iid)
        if ref: rs[ref].append(iid)
        ts[tt].append(iid)
    ut[t]={'dup_statement_groups':sum(1 for v in st.values() if len(v)>1),'dup_statement_rows':sum(len(v) for v in st.values() if len(v)>1),
           'dup_reference_solution_groups':sum(1 for v in rs.values() if len(v)>1),'dup_reference_solution_rows':sum(len(v) for v in rs.values() if len(v)>1),
           'dup_tests_groups':sum(1 for v in ts.values() if len(v)>1),'dup_tests_rows':sum(len(v) for v in ts.values() if len(v)>1),
           'dup_statement_AND_tests_rows':sum(v for v in collections.Counter((s,tt) for _,s,_,tt in ks).values() if v>1),
           'dup_statement_examples':[v[:4] for v in st.values() if len(v)>1][:5],
           'dup_reference_examples':[v[:4] for v in rs.values() if len(v)>1][:5],
           'dup_tests_examples':[v[:4] for v in ts.values() if len(v)>1][:5]}
core=keys['t15_code_unittest']; ext=keys['t15_code_unittest_ext']
cs={s:i for i,s,_,_ in core}; cr={r:i for i,_,r,_ in core if r}; ct={t:i for i,_,_,t in core}
shared_st=[(cs[s],i) for i,s,_,_ in ext if s in cs]; shared_rs=[(cr[r],i) for i,_,r,_ in ext if r in cr]; shared_ts=[(ct[t],i) for i,_,_,t in ext if t in ct]
ut['cross_core_ext']={'ext_rows_with_statement_in_core':len(shared_st),'ext_rows_with_reference_in_core':len(shared_rs),'ext_rows_with_tests_in_core':len(shared_ts),
    'shared_statement_example_ids':shared_st[:5],'shared_reference_example_ids':shared_rs[:5],'shared_tests_example_ids':shared_ts[:5]}
S6['unittest']=ut
S6['seconds_total']=round(time.time()-t0,1)
json.dump(S6,open(f'{OUT}/_dupes_part.json','w'),indent=1,ensure_ascii=False,default=str)
print('dupes done',S6['seconds_total'])

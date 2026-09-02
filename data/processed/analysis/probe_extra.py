import pyarrow.parquet as pq, json, re, collections, hashlib
P='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/parquet'; D='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/index'
def load(p): return [json.loads(l) for l in open(p,encoding='utf-8')]
O={}
UF={}
for t in ('t15_code_unittest','t15_code_unittest_ext'):
    for r in load(f'{D}/{t}.full.jsonl'): UF[r['instance_id']]=r
UL={}
for t in ('t15_code_unittest','t15_code_unittest_ext'):
    for r in load(f'{D}/{t}.learnable.jsonl'): UL[r['instance_id']]=r
groups=json.load(open('/scratch/r919a03/code_verl_test/data/processed/analysis/_probe_followup.json'))['identical_prompt_groups_across_all_parquet']
gg=[]
for g in groups:
    ids=[i for _,i in g]
    tests=[json.dumps(UF[i]['tests'],sort_keys=True) for i in ids]
    refs=[re.sub(r'\s+',' ',UF[i]['reference_solution'] or '') for i in ids]
    gg.append({'ids':ids,'files':[n for n,_ in g],'same_tests':len(set(tests))==1,'same_reference':len(set(refs))==1,'n_tests':[len(UF[i]['tests']) for i in ids],'p_hat':[UL[i]['p_hat'] for i in ids]})
O['identical_prompt_groups_detail']=gg
# p_hat quantisation vs source
O['p_hat_source_x_p_hat_core']=dict(collections.Counter(f"{r['p_hat_source']}|{r['p_hat']}" for r in UL.values() if r['p_hat'] is not None))
# SWE snapshot file counts
F={r['instance_id']:r for r in load(f'{D}/t15_code_swesmith.full.jsonl')}
rows=[r for n in ('t15_code_rl_core_train','t15_code_rl_core_val') for r in pq.read_table(f'{P}/{n}.parquet').to_pylist() if r['data_source']=='t15_repo_patch']
nf=[]; byv=collections.defaultdict(list)
for r in rows:
    m=re.search(r'Files included \((\d+)\)',r['prompt'][1]['content']); n=int(m.group(1)); nf.append(n); byv[r['_source_meta']['variant_type']].append(n)
def dist(xs):
    xs=sorted(xs); n=len(xs); q=lambda p: xs[min(n-1,int(round(p*(n-1))))]
    return {'n':n,'min':xs[0],'median':q(.5),'p90':q(.9),'max':xs[-1]}
O['swe_snapshot_files_included_dist']=dist(nf); O['swe_snapshot_files_by_variant']={k:dist(v) for k,v in byv.items()}
O['swe_instances_per_repo_full']=dict(collections.Counter(r['repo'] for r in F.values()))
# inflect: all 40 gold touch inflect/__init__.py?
inf=[r for r in F.values() if 'inflect' in r['repo']]
O['inflect_n']=len(inf); O['inflect_gold_touches_init']=sum(1 for r in inf if 'diff --git a/inflect/__init__.py' in r['gold_patch'])
O['inflect_f2p_file_is_init']=sum(1 for r in inf if any(x.startswith('inflect/__init__.py::') for x in r['FAIL_TO_PASS']))
O['inflect_f2p_files']=dict(collections.Counter(x.split('::')[0] for r in inf for x in r['FAIL_TO_PASS']))
# R2E total f2p ids
R2=load(f'{D}/t15_code_r2e.full.jsonl'); O['r2e_total_f2p_ids']=sum(len(r['FAIL_TO_PASS']) for r in R2); O['r2e_f2p_ids_with_::']=sum(1 for r in R2 for x in r['FAIL_TO_PASS'] if '::' in x)
# SWE: problem statement mentions the test names / file paths of gold? (artifact leak proxy)
O['swe_issue_mentions_gold_path']=sum(1 for r in F.values() if any(p in r['problem_statement'] for p in re.findall(r'^diff --git a/(\S+) b/',r['gold_patch'],re.M)))
# core unittest: which ids in val share statement with train (exact, normalized)
tr=pq.read_table(f'{P}/t15_code_rl_core_train.parquet').to_pylist(); va=pq.read_table(f'{P}/t15_code_rl_core_val.parquet').to_pylist()
ns=lambda s: re.sub(r'\s+',' ',s.strip())
trs={ns(r['prompt'][1]['content']):r['extra_info']['instance_id'] for r in tr}
O['val_rows_with_user_msg_in_train']=[(r['extra_info']['instance_id'],trs[ns(r['prompt'][1]['content'])]) for r in va if ns(r['prompt'][1]['content']) in trs]
json.dump(O,open('/scratch/r919a03/code_verl_test/data/processed/analysis/_probe_extra.json','w'),indent=1,ensure_ascii=False)
print(json.dumps(O,indent=1,ensure_ascii=False)[:9000])

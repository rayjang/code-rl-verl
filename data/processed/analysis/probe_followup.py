import pyarrow.parquet as pq, json, re, collections, hashlib
P='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/parquet'; D='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/index'
def load(p): return [json.loads(l) for l in open(p,encoding='utf-8')]
O={}
# 1. pytest n_tests == number of def test_ functions?
for t in ('t15_code_unittest','t15_code_unittest_ext'):
    F=load(f'{D}/{t}.full.jsonl'); L={r['instance_id']:r for r in load(f'{D}/{t}.learnable.jsonl')}
    c=collections.Counter()
    for r in F:
        n=L[r['instance_id']]['n_tests']
        if r['harness']=='pytest':
            k=sum(len(re.findall(r'^\s*def\s+test\w*\s*\(',x.get('assertion') or '',re.M)) for x in r['tests'])
            c['pytest|n_tests==def_test_count' if n==k else 'pytest|n_tests!=def_test_count']+=1
        else:
            c[f"{r['harness']}|n_tests==len(tests)" if n==len(r['tests']) else f"{r['harness']}|n_tests!=len(tests)"]+=1
    O[f'n_tests_semantics|{t}']=dict(c)
# 3. SWE gold vs judge test gate: categorize the 47
_TESTISH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$|(^|/)(pytest|tox|setup)\.cfg$|(^|/)pyproject\.toml$")
F=load(f'{D}/t15_code_swesmith.full.jsonl'); L={r['instance_id']:r for r in load(f'{D}/t15_code_swesmith.learnable.jsonl')}
cat=collections.Counter(); ex=collections.defaultdict(list); byrepo=collections.Counter(); doctest_ids=0
for r in F:
    paths=sorted(set(re.findall(r'^diff --git a/(\S+) b/',r['gold_patch'],re.M)))
    tf={x.split('::')[0] for x in r['FAIL_TO_PASS']+r['PASS_TO_PASS'][:30]}
    a=[p for p in paths if p in tf]; b=[p for p in paths if _TESTISH.search(p)]
    if a or b:
        k=('gold_path_is_F2P/P2P_test_file' if a else '')+('+' if a and b else '')+('gold_path_matches_TESTISH_regex' if b else '')
        cat[k]+=1; ex[k].append((r['instance_id'],a or b)); byrepo[r['repo']]+=1
    if any(not x.split('::')[0].endswith('.py') or '::' in x and re.search(r'\.py::[A-Za-z_][\w.]*$',x) is None for x in r['FAIL_TO_PASS']): pass
O['swe_gold_would_be_rejected_by_touches_tests']={'total':sum(cat.values()),'by_category':dict(cat),'by_repo':dict(byrepo),'examples':{k:v[:6] for k,v in ex.items()}}
# F2P ids pointing at non-test source files (doctests): file part not matching TESTISH
src_f2p=collections.Counter()
for r in F:
    for x in r['FAIL_TO_PASS']:
        f=x.split('::')[0]
        if not _TESTISH.search(f): src_f2p[r['repo']]+=1
O['swe_f2p_ids_whose_file_is_not_testish_by_repo']=dict(src_f2p)
O['swe_f2p_nontestish_examples']=[x for r in F for x in r['FAIL_TO_PASS'] if not _TESTISH.search(x.split('::')[0])][:8]
# 6. judge_verdict crosstabs
Ls=list(L.values())
O['verdict_x_scope_match']=dict(collections.Counter(f"{r['judge_verdict']}|scope_match={r['quality']['scope_match']}" for r in Ls))
O['verdict_x_test_alignment']=dict(collections.Counter(f"{r['judge_verdict']}|test_alignment={r['quality']['test_alignment']}" for r in Ls))
O['verdict_x_issue_quality']=dict(collections.Counter(f"{r['judge_verdict']}|issue_quality={r['quality']['issue_quality']}" for r in Ls))
O['verdict_x_judge_difficulty']=dict(collections.Counter(f"{r['judge_verdict']}|{r['judge_difficulty']}" for r in Ls))
O['verdict_x_bugkind']=dict(collections.Counter(f"{r['judge_verdict']}|{re.sub(r'__.*$','',r['instance_id'].split('.')[2]) if r['instance_id'].count('.')>=2 else 'pr'}" for r in Ls).most_common(20))
O['fix_verdict_rows']=[(r['instance_id'],r['quality']) for r in Ls if r['judge_verdict']=='fix']
O['ko_translation_null_iff_ps_ko_identical']=sum(1 for r in F if (L[r['instance_id']]['quality']['ko_translation'] is None)==(r['problem_statement']==r['problem_statement_ko']))
# parquet-side probes
rows={n:pq.read_table(f'{P}/{n}.parquet').to_pylist() for n in ('t15_code_rl_core_train','t15_code_rl_core_val','t15_code_rl_extended_train')}
allr=[(n,r) for n,rs in rows.items() for r in rs]
# 13. image_name for unittest rows
O['extra_info.image_name_values_unittest']=dict(collections.Counter(r['extra_info']['image_name'] for n,r in allr if r['data_source']=='t15_unittest_impl'))
O['extra_info.image_name_values_swe_sample']=dict(collections.Counter(r['extra_info']['image_name'] for n,r in allr if r['data_source']=='t15_repo_patch').most_common(3))
# 14. rubrics: all reward false? criteria distribution by source
O['rubric_tag_reward_true_count']=sum(1 for n,r in allr for rb in (r['extra_info']['rubrics'] or []) if rb['tags']['reward'])
O['rubric_tag_judge_true_count']=sum(1 for n,r in allr for rb in (r['extra_info']['rubrics'] or []) if rb['tags']['judge'])
O['rubric_tags_verify_values']=dict(collections.Counter(rb['tags']['verify'] for n,r in allr for rb in (r['extra_info']['rubrics'] or [])))
O['rubric_criteria_unittest_top']=dict(collections.Counter(rb['criterion'] for n,r in allr if r['data_source']=='t15_unittest_impl' for rb in (r['extra_info']['rubrics'] or [])).most_common(12))
O['rubric_criteria_swe_top']=dict(collections.Counter(re.sub(r'test_\w+(, )?','<T>',rb['criterion'])[:90] for n,r in allr if r['data_source']=='t15_repo_patch' for rb in (r['extra_info']['rubrics'] or [])).most_common(10))
# 15/unittest prompt == statement? entry point in prompt?
UF={}
for t in ('t15_code_unittest','t15_code_unittest_ext'):
    for r in load(f'{D}/{t}.full.jsonl'): UF[r['instance_id']]=r
c=collections.Counter(); ex2={}
for n,r in allr:
    if r['data_source']!='t15_unittest_impl': continue
    f=UF[r['extra_info']['instance_id']]; u=r['prompt'][1]['content']; s=r['prompt'][0]['content']
    c['user==statement' if u==f['statement'] else ('user.strip()==statement.strip()' if u.strip()==f['statement'].strip() else 'user!=statement')]+=1
    if u!=f['statement'] and u.strip()!=f['statement'].strip(): ex2.setdefault('user_ne_statement',(r['extra_info']['instance_id'],u[:300],'|||',f['statement'][:300]))
    if f['entry_point']:
        c['entry_point_in_user_msg' if f['entry_point'] in u else 'entry_point_NOT_in_user_msg']+=1
    c['sys='+hashlib.md5(s.encode()).hexdigest()[:6]]+=1
    c['prompt_mentions_stdin' if ('stdin' in u.lower() or 'input format' in u.lower()) else 'no_stdin_mention']+= (f['harness']=='stdio')
O['unittest_prompt_vs_statement']=dict(c); O['unittest_prompt_examples']=ex2
# pytest example: show the test file for one pytest instance whose statement lacks the name
r=UF['kodcode-a154fafde017']; O['pytest_example_test_file']=r['tests'][0]['assertion'][:900]; O['pytest_example_entry_point']=r['entry_point']; O['pytest_example_ref_head']=r['reference_solution'][:300]
r=UF['opencodeinstruct-8fe7c2210ce5']; O['function_example_statement']=r['statement']; O['function_example_tests']=[x['assertion'] for x in r['tests'][:3]]
# 10/11. duplicates across train/val and within
h=lambda r: hashlib.md5(json.dumps(r['prompt']).encode()).hexdigest()
tv=collections.defaultdict(list)
for n,r in allr: tv[h(r)].append((n,r['extra_info']['instance_id']))
O['identical_prompt_groups_across_all_parquet']=[v for v in tv.values() if len(v)>1]
# 12. SWE prompt token estimate vs verl max_prompt_length
sw=[(n,sum(len(m['content']) for m in r['prompt'])/3.5,r['_source_meta']['variant_type']) for n,r in allr if r['data_source']=='t15_repo_patch']
O['swe_rows_est_tokens_gt_32768_by_variant']=dict(collections.Counter(v for n,tk,v in sw if tk>32768))
O['swe_rows_est_tokens_le_32768_by_variant']=dict(collections.Counter(v for n,tk,v in sw if tk<=32768))
# 2. SWE user-message language vs lang; test file shown; p2p list truncated to 30 in reward
O['swe_lang_values_parquet']=dict(collections.Counter(r['extra_info']['lang'] for n,r in allr if r['data_source']=='t15_repo_patch'))
json.dump(O,open('/scratch/r919a03/code_verl_test/data/processed/analysis/_probe_followup.json','w'),indent=1,ensure_ascii=False,default=str)
print(json.dumps(O,indent=1,ensure_ascii=False,default=str)[:14000])

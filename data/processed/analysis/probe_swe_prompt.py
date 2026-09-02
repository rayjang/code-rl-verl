import pyarrow.parquet as pq, json, re, collections
P='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/parquet'; D='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/index'
full={}
for l in open(f'{D}/t15_code_swesmith.full.jsonl',encoding='utf-8'):
    d=json.loads(l); full[d['instance_id']]=d
rows=[r for f in ('t15_code_rl_core_train','t15_code_rl_core_val') for r in pq.read_table(f'{P}/{f}.parquet').to_pylist() if r['data_source']=='t15_repo_patch']
C=collections.Counter(); ex={}
sysmsgs=collections.Counter(); tails=collections.Counter()
for r in rows:
    iid=r['extra_info']['instance_id']; f=full[iid]; sysm=r['prompt'][0]['content']; u=r['prompt'][1]['content']
    sysmsgs[sysm[:120]]+=1
    lang=r['extra_info']['lang']
    C[f'lang={lang}']+=1
    # which problem statement is embedded
    if f['problem_statement'] in u and f['problem_statement_ko'] in u and f['problem_statement']!=f['problem_statement_ko']: C['both_ps_in_user']+=1
    elif f['problem_statement_ko'] in u and f['problem_statement']!=f['problem_statement_ko']: C[f'ko_ps_in_user|lang={lang}']+=1
    elif f['problem_statement'] in u: C[f'en_ps_in_user|lang={lang}']+=1
    else: C['ps_not_found_verbatim']+=1; ex.setdefault('ps_not_found',iid)
    # task tail
    m=re.search(r'\n# Task\n(.*)$',u,re.S)
    tail=m.group(1).strip() if m else 'NO_TASK_SECTION'
    tails[re.sub(r'\s+',' ',tail)[:160]]+=1
    C['sys_mentions_SEARCH_REPLACE']+= '<<<<<<< SEARCH' in sysm
    C['user_task_mentions_diff_block']+= ('```diff' in tail)
    C['user_task_mentions_SEARCH_REPLACE']+= ('SEARCH' in tail)
    C['user_msg_has_hangul']+= bool(re.search(r'[가-힣]',u)); C['sys_msg_has_hangul']+= bool(re.search(r'[가-힣]',sysm))
    # snapshot files
    m2=re.search(r'# Repository snapshot\nFiles included \((\d+)\):\n```\n(.*?)\n```',u,re.S)
    if not m2: C['no_snapshot_list']+=1; continue
    listed=set(m2.group(2).split('\n')); nlisted=int(m2.group(1))
    shown=set(re.findall(r'^\[FILE\] (\S+)$',u,re.M))
    C['listed_count_matches_FILE_blocks']+= (len(shown)==nlisted==len(listed))
    gold_paths=set(re.findall(r'^diff --git a/(\S+) b/',f['gold_patch'],re.M))
    if gold_paths<=shown: C['gold_files_all_in_snapshot']+=1
    else: C['gold_files_MISSING_from_snapshot']+=1; ex.setdefault('gold_missing',(iid,sorted(gold_paths-shown)))
    tf={t.split('::')[0] for t in f['FAIL_TO_PASS']}
    if tf & shown: C['f2p_test_file_shown_in_snapshot']+=1; ex.setdefault('f2p_shown',(iid,sorted(tf&shown)[:3]))
    else: C['f2p_test_file_NOT_in_snapshot']+=1
    testish=[p for p in shown if re.search(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$',p)]
    C['snapshot_contains_any_testish_file']+= bool(testish)
    # are buggy_files contents identical to snapshot file contents?
    bf=json.loads(f['buggy_files'])
    for p,src in bf.items():
        blk=re.search(r'^\[FILE\] '+re.escape(p)+r'\n```(?:python)?\n(.*?)\n```\n',u,re.S|re.M)
        if blk is None: C['buggy_file_not_shown']+=1
        elif blk.group(1).strip()==src.strip(): C['buggy_file_content_matches_snapshot']+=1
        else: C['buggy_file_content_DIFFERS_from_snapshot']+=1; ex.setdefault('bf_differs',(iid,p))
    # F2P test names leak in issue text?
    names={t.split('::')[-1].split('[')[0] for t in f['FAIL_TO_PASS']}
    if any(n in f['problem_statement'] for n in names): C['f2p_test_name_appears_in_issue']+=1
print(json.dumps({'n_rows':len(rows),'counts':dict(C),'examples':ex,'system_prompt_variants':dict(sysmsgs),'task_tail_variants':dict(tails.most_common(6))},indent=1,ensure_ascii=False))

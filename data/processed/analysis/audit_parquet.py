#!/usr/bin/env python3
"""Parquet-side audit (section 5 + parquet-side section 7)."""
import pyarrow.parquet as pq, json, collections, os, re, glob, hashlib
P='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/parquet'
D='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/index'
OUT='/scratch/r919a03/code_verl_test/data/processed/analysis'
def dist(xs):
    xs=sorted(xs); n=len(xs)
    if not n: return {}
    q=lambda p: xs[min(n-1,int(round(p*(n-1))))]
    return {'n':n,'min':xs[0],'p10':q(.1),'p25':q(.25),'median':q(.5),'p75':q(.75),'p90':q(.9),'p99':q(.99),'max':xs[-1],'mean':round(sum(xs)/n,1)}
def vc(xs): return dict(collections.Counter(xs).most_common())
learn={}
for t in ('t15_code_swesmith','t15_code_unittest','t15_code_unittest_ext','t15_code_r2e'):
    for l in open(f'{D}/{t}.learnable.jsonl',encoding='utf-8'):
        d=json.loads(l); d['_tag']=t; learn[d['instance_id']]=d
R={}; per={}; allrows={}
for f in sorted(glob.glob(P+'/*.parquet')):
    name=os.path.basename(f).replace('.parquet','')
    t=pq.read_table(f)
    rows=t.to_pylist()
    allrows[name]=rows
    S={'file':f,'rows':len(rows),'columns':t.column_names,'schema':str(t.schema)}
    S['data_source']=vc(r['data_source'] for r in rows); S['ability']=vc(r['ability'] for r in rows)
    S['reward_model.style']=vc(r['reward_model']['style'] for r in rows)
    S['reward_model.ground_truth_nonempty']=sum(1 for r in rows if r['reward_model']['ground_truth'])
    ei=[r['extra_info'] for r in rows]
    for k in ('split','task_tag','rubric_profile','variant_type','exec_backend','harness','p_hat_source','difficulty_band','lang','context_bucket','difficulty_0_10'):
        S[f'extra_info.{k}']=vc(str(e.get(k)) for e in ei)
    S['extra_info.p_hat_nonnull']=sum(1 for e in ei if e.get('p_hat') is not None)
    S['extra_info.p_hat_values']=vc(str(e.get('p_hat')) for e in ei)
    S['extra_info.image_name_nonempty']=sum(1 for e in ei if e.get('image_name')); S['extra_info.repo_nonempty']=sum(1 for e in ei if e.get('repo'))
    S['extra_info.repo_values_by_source']={ds:vc(e.get('repo') for e,r in zip(ei,rows) if r['data_source']==ds) for ds in S['data_source']}
    S['extra_info.rubrics_len']=vc(len(e.get('rubrics') or []) for e in ei)
    S['extra_info.rubrics_example']=next((e['rubrics'] for e in ei if e.get('rubrics')),None)
    S['extra_info.n_tests_dist_unittest']=dist([e['n_tests'] for e,r in zip(ei,rows) if r['data_source']=='t15_unittest_impl' and e.get('n_tests') is not None])
    S['extra_info.n_f2p_dist_swe']=dist([e['n_f2p'] for e,r in zip(ei,rows) if r['data_source']=='t15_repo_patch' and e.get('n_f2p') is not None])
    S['extra_info.n_f2p_dist_unittest']=dist([e['n_f2p'] for e,r in zip(ei,rows) if r['data_source']=='t15_unittest_impl' and e.get('n_f2p') is not None])
    S['_source_meta.dataset']=vc(r['_source_meta']['dataset'] for r in rows); S['_source_meta.variant_type']=vc(r['_source_meta']['variant_type'] for r in rows)
    # id consistency
    S['index==instance_id']=sum(1 for r in rows if r['extra_info']['index']==r['extra_info']['instance_id'])
    S['ground_truth==instance_id']=sum(1 for r in rows if r['reward_model']['ground_truth']==r['extra_info']['instance_id'])
    S['ground_truth!=instance_id_examples']=[(r['reward_model']['ground_truth'],r['extra_info']['instance_id']) for r in rows if r['reward_model']['ground_truth']!=r['extra_info']['instance_id']][:5]
    ids=[r['extra_info']['instance_id'] for r in rows]
    S['unique_instance_id']=len(set(ids)); S['dup_instance_ids']=[k for k,v in collections.Counter(ids).items() if v>1][:10]
    # join to index
    j=collections.Counter(learn[i]['_tag'] if i in learn else 'MISSING' for i in ids)
    S['join_to_index_by_instance_id']=dict(j)
    S['missing_join_examples']=[i for i in ids if i not in learn][:10]
    mism=collections.Counter()
    for r in rows:
        l=learn.get(r['extra_info']['instance_id'])
        if not l: continue
        e=r['extra_info']
        for k in ('p_hat','p_hat_source','difficulty_band','difficulty_0_10','n_f2p','n_p2p','n_tests','harness','exec_backend','lang','image_name','repo','context_bucket'):
            if k in l and e.get(k)!=l.get(k): mism[k]+=1
        if r['data_source']!=l['data_source']: mism['data_source']+=1
        if r['ability']!=l['ability']: mism['ability']+=1
        if e.get('subset') is not None and e.get('subset')!=l.get('subset'): mism['subset']+=1
    S['extra_info_vs_learnable_mismatch_counts']=dict(mism)
    S['judge_verdict_of_swe_rows']=vc(learn[i].get('judge_verdict') for i in ids if i in learn and learn[i]['_tag']=='t15_code_swesmith')
    # prompt
    roles=vc(json.dumps([m['role'] for m in r['prompt']]) for r in rows); S['prompt_role_sequences']=roles
    S['prompt_n_messages']=vc(len(r['prompt']) for r in rows)
    lens=[sum(len(m['content'] or '') for m in r['prompt']) for r in rows]
    S['prompt_len_chars_dist']=dist(lens); S['prompt_len_tokens_est_dist']={k:(round(v/3.5) if isinstance(v,(int,float)) and k!='n' else v) for k,v in dist(lens).items()}
    S['prompt_len_chars_dist_by_source']={ds:dist([l for l,r in zip(lens,rows) if r['data_source']==ds]) for ds in S['data_source']}
    S['prompt_len_chars_dist_by_variant']={v:dist([l for l,r in zip(lens,rows) if r['_source_meta']['variant_type']==v]) for v in S['_source_meta.variant_type']}
    S['prompt_gt_30k_tokens_est']=sum(1 for l in lens if l/3.5>30000); S['prompt_gt_32768_tokens_est']=sum(1 for l in lens if l/3.5>32768)
    S['prompt_gt_30k_tokens_by_variant']=vc(r['_source_meta']['variant_type'] for l,r in zip(lens,rows) if l/3.5>30000)
    S['prompt_empty_rows']=sum(1 for r in rows if not r['prompt'] or any(not (m['content'] or '').strip() for m in r['prompt']))
    S['system_prompt_variants']=vc(hashlib.md5((r['prompt'][0]['content'] or '').encode()).hexdigest()[:8]+'|'+r['data_source'] for r in rows if r['prompt'] and r['prompt'][0]['role']=='system')
    S['prompt_text_dup_groups']=sum(1 for v in collections.Counter(hashlib.md5(json.dumps(r['prompt']).encode()).hexdigest() for r in rows).values() if v>1)
    per[name]=S
    for r,l in zip(rows,lens):
        allrows.setdefault('_len',{})[r['extra_info']['instance_id']]=(l,name,r['_source_meta']['variant_type'],r['extra_info'].get('lang'),r['extra_info'].get('repo') or r['extra_info'].get('image_name') or '')
R['per_file']=per
# cross-file
tr=allrows['t15_code_rl_core_train']; va=allrows['t15_code_rl_core_val']; ex=allrows['t15_code_rl_extended_train']
ids=lambda rs:{r['extra_info']['instance_id'] for r in rs}
R['split']={'core_train_ids':len(ids(tr)),'core_val_ids':len(ids(va)),'ext_ids':len(ids(ex)),
    'train&val':len(ids(tr)&ids(va)),'train&ext':len(ids(tr)&ids(ex)),'val&ext':len(ids(va)&ids(ex)),
    'val_data_source':vc(r['data_source'] for r in va),'val_variant':vc(r['_source_meta']['variant_type'] for r in va),
    'val_split_field':vc(r['extra_info']['split'] for r in va),'train_split_field':vc(r['extra_info']['split'] for r in tr),'ext_split_field':vc(r['extra_info']['split'] for r in ex)}
def repos(rs): return {r['extra_info'].get('repo') for r in rs if r['data_source']=='t15_repo_patch'}
R['split']['swe_repos_train']=sorted(repos(tr)); R['split']['swe_repos_val']=sorted(repos(va)); R['split']['swe_repo_overlap_train_val']=sorted(repos(tr)&repos(va))
R['split']['swe_val_repo_counts']=vc(r['extra_info'].get('repo') for r in va if r['data_source']=='t15_repo_patch')
R['split']['swe_train_repo_counts']=vc(r['extra_info'].get('repo') for r in tr if r['data_source']=='t15_repo_patch')
def ptxt(rs): return {hashlib.md5(json.dumps(r['prompt']).encode()).hexdigest() for r in rs}
R['split']['train&val_identical_prompt_text']=len(ptxt(tr)&ptxt(va))
def ustmt(rs): return {re.sub(r'\s+',' ',r['prompt'][-1]['content'].strip()) for r in rs if r['data_source']=='t15_unittest_impl'}
R['split']['train&val_identical_unittest_user_msg']=len(ustmt(tr)&ustmt(va))
R['split']['val_frac_of_core']=round(len(va)/(len(va)+len(tr)),4)
R['split']['val_per_source_frac']={ds:round(sum(1 for r in va if r['data_source']==ds)/max(1,sum(1 for r in tr+va if r['data_source']==ds)),4) for ds in ('t15_repo_patch','t15_unittest_impl')}
# reverse join: index rows not in any parquet
allp=ids(tr)|ids(va)|ids(ex)
R['index_rows_not_in_parquet']=vc(l['_tag'] for i,l in learn.items() if i not in allp)
R['index_rows_not_in_parquet_examples']=[i for i,l in learn.items() if i not in allp and l['_tag']!='t15_code_r2e'][:10]
# example prompts: shortest SWE and a median-length unittest, plus SWE one from core (both roles)
swe=[r for r in tr if r['data_source']=='t15_repo_patch']; ut=[r for r in tr if r['data_source']=='t15_unittest_impl']
swe.sort(key=lambda r:sum(len(m['content']) for m in r['prompt'])); ut.sort(key=lambda r:sum(len(m['content']) for m in r['prompt']))
ex_swe=swe[0]; ex_ut=ut[len(ut)//2]
for nm,r in (('example_prompt_swe',ex_swe),('example_prompt_unittest',ex_ut)):
    with open(f'{OUT}/{nm}.txt','w',encoding='utf-8') as f:
        f.write(f"# instance_id={r['extra_info']['instance_id']} data_source={r['data_source']} variant={r['_source_meta']['variant_type']} total_chars={sum(len(m['content']) for m in r['prompt'])}\n")
        for m in r['prompt']:
            f.write(f"\n===== ROLE: {m['role']} (chars={len(m['content'])}) =====\n{m['content']}\n")
    R.setdefault('examples',{})[nm]={'instance_id':r['extra_info']['instance_id'],'file':f'{OUT}/{nm}.txt','roles':[m['role'] for m in r['prompt']],'chars':[len(m['content']) for m in r['prompt']]}
# also dump one pytest-harness unittest prompt and a stdio one if present
for h in ('pytest','stdio','function'):
    c=[r for r in tr+ex if r['data_source']=='t15_unittest_impl' and r['extra_info'].get('harness')==h]
    if c:
        c.sort(key=lambda r:sum(len(m['content']) for m in r['prompt'])); r=c[len(c)//2]
        with open(f'{OUT}/example_prompt_unittest_{h}.txt','w',encoding='utf-8') as f:
            f.write(f"# instance_id={r['extra_info']['instance_id']} harness={h} total_chars={sum(len(m['content']) for m in r['prompt'])}\n")
            for m in r['prompt']: f.write(f"\n===== ROLE: {m['role']} (chars={len(m['content'])}) =====\n{m['content']}\n")
json.dump(allrows['_len'],open(f'{OUT}/_prompt_len.json','w'))
json.dump(R,open(f'{OUT}/_parquet_part.json','w'),indent=1,ensure_ascii=False,default=str)
print('parquet audit done')

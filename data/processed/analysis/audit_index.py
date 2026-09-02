#!/usr/bin/env python3
"""Index-side audit (sections 1,2,3,4,6,7-index). Read-only on sources."""
import json, glob, os, re, statistics, collections, difflib, itertools, hashlib, sys
D='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/index'
OUT='/scratch/r919a03/code_verl_test/data/processed/analysis'
TAGS=['t15_code_swesmith','t15_code_unittest','t15_code_unittest_ext','t15_code_r2e']

def load(p): return [json.loads(l) for l in open(p,encoding='utf-8')]
def dist(xs):
    xs=sorted(xs)
    if not xs: return {}
    n=len(xs)
    def q(p): return xs[min(n-1,int(round(p*(n-1))))]
    return {'n':n,'min':xs[0],'p10':q(.1),'p25':q(.25),'median':q(.5),'p75':q(.75),'p90':q(.9),'p99':q(.99),'max':xs[-1],'mean':round(sum(xs)/n,3)}
def vc(xs): return dict(collections.Counter(xs).most_common())
def empty(v): return v is None or v=='' or v==[] or v=={}

R={}
full={t:load(f'{D}/{t}.full.jsonl') for t in TAGS}
learn={t:load(f'{D}/{t}.learnable.jsonl') for t in TAGS}

# ---------- Section 1: schema / uniqueness / cross-file dupes ----------
S1={}
for t in TAGS:
    for kind,rows in (('full',full[t]),('learnable',learn[t])):
        keys=collections.OrderedDict()
        for r in rows:
            for k,v in r.items():
                e=keys.setdefault(k,{'types':collections.Counter(),'nonnull':0,'nonempty':0})
                e['types'][type(v).__name__]+=1
                if v is not None: e['nonnull']+=1
                if not empty(v): e['nonempty']+=1
        ids=[r['instance_id'] for r in rows]
        dup=[k for k,c in collections.Counter(ids).items() if c>1]
        S1[f'{t}.{kind}']={'file':f'{D}/{t}.{kind}.jsonl','rows':len(rows),
            'fields':{k:{'types':dict(e['types']),'pct_nonnull':round(100*e['nonnull']/len(rows),2),'pct_nonempty':round(100*e['nonempty']/len(rows),2)} for k,e in keys.items()},
            'n_unique_instance_id':len(set(ids)),'duplicate_ids_within_file':dup}
    # full vs learnable alignment
    fi=[r['instance_id'] for r in full[t]]; li=[r['instance_id'] for r in learn[t]]
    S1[f'{t}.full_vs_learnable']={'same_id_set':set(fi)==set(li),'same_order':fi==li,'only_in_full':sorted(set(fi)-set(li))[:20],'only_in_learnable':sorted(set(li)-set(fi))[:20]}
cross={}
for a,b in itertools.combinations(TAGS,2):
    inter=set(r['instance_id'] for r in full[a])&set(r['instance_id'] for r in full[b])
    cross[f'{a}&{b}']={'n':len(inter),'examples':sorted(inter)[:10]}
S1['cross_file_instance_id_overlap']=cross
R['section1_index_files']=S1

# ---------- Section 2: SWE ----------
_TESTISH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|"
                      r"(^|/)conftest\.py$|(^|/)(pytest|tox|setup)\.cfg$|(^|/)pyproject\.toml$")
_DIFF_PATH = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
def touched_paths(patch_str):
    out=[]
    for m in _DIFF_PATH.finditer(patch_str or ''): out+=[m.group(1),m.group(2)]
    for line in (patch_str or '').split('\n'):
        if line.startswith('--- a/') or line.startswith('+++ b/'): out.append(line[6:].strip())
    return sorted({p for p in out if p and p!='/dev/null'})
IID_RE=re.compile(r'^(?P<owner>.+?)__(?P<repo>.+)\.(?P<commit>[0-9a-f]{8})\.(?P<kind>[A-Za-z_]+?)__(?P<hash>[A-Za-z0-9_]+)$')
manifest=json.load(open(f'{D}/images_manifest.json'))
manifest_imgs={x['image'] for x in manifest['images']}
img_swe=set(l.strip() for l in open(f'{D}/images_t15_code_swesmith.txt') if l.strip())
img_r2e=set(l.strip() for l in open(f'{D}/images_t15_code_r2e.txt') if l.strip())

def swe_section(t):
    F=full[t]; L={r['instance_id']:r for r in learn[t]}
    S={}
    S['rows']=len(F)
    S['all_keys_full']=sorted({k for r in F for k in r}); S['all_keys_learnable']=sorted({k for r in learn[t] for k in r})
    S['base_commit_like_fields_present']=sorted({k for r in F+learn[t] for k in r if re.search(r'commit|branch|base|sha|ref|version',k,re.I)})
    S['test_patch_field_present']=any('test_patch' in r for r in F)
    repos=[r['repo'] for r in F]; imgs=[r['image_name'] for r in F]
    S['n_distinct_repos']=len(set(repos)); S['n_distinct_images']=len(set(imgs))
    S['instances_per_repo']=vc(repos); S['instances_per_image']=vc(imgs)
    S['repo_to_image_is_1to1']=len({(r['repo'],r['image_name']) for r in F})==len(set(repos))==len(set(imgs))
    S['images_not_in_manifest']=sorted(set(imgs)-manifest_imgs)
    S['images_not_in_images_txt']=sorted(set(imgs)-(img_swe if t=='t15_code_swesmith' else img_r2e))
    S['learnable_image_name_matches_full']=all(L[r['instance_id']]['image_name']==r['image_name'] for r in F)
    S['sif_paths_examples']=sorted({L[r['instance_id']]['sif'] for r in F})[:3]
    S['sif_basename_matches_image']=all(os.path.basename(L[r['instance_id']]['sif'])==r['image_name'].rsplit('/',1)[-1].replace(':','_')+'.sif' for r in F)
    f2p=[len(r['FAIL_TO_PASS']) for r in F]; p2p=[len(r['PASS_TO_PASS']) for r in F]
    S['f2p_count_dist']=dist(f2p); S['p2p_count_dist']=dist(p2p)
    S['zero_f2p_instances']=[r['instance_id'] for r in F if len(r['FAIL_TO_PASS'])==0]
    S['zero_p2p_instances']=[r['instance_id'] for r in F if len(r['PASS_TO_PASS'])==0]
    S['learnable_n_f2p_mismatch']=[r['instance_id'] for r in F if L[r['instance_id']]['n_f2p']!=len(r['FAIL_TO_PASS'])]
    S['learnable_n_p2p_mismatch']=[r['instance_id'] for r in F if L[r['instance_id']]['n_p2p']!=len(r['PASS_TO_PASS'])]
    S['f2p_ids_without_double_colon']=sum(1 for r in F for x in r['FAIL_TO_PASS'] if '::' not in x)
    S['p2p_ids_without_double_colon']=sum(1 for r in F for x in r['PASS_TO_PASS'] if '::' not in x)
    S['f2p_truncated_param_ids']=sum(1 for r in F for x in r['FAIL_TO_PASS'] if '[' in x and not x.endswith(']'))
    S['f2p_in_p2p_overlap_instances']=sum(1 for r in F if set(r['FAIL_TO_PASS'])&set(r['PASS_TO_PASS']))
    S['f2p_duplicate_ids_instances']=sum(1 for r in F if len(set(r['FAIL_TO_PASS']))!=len(r['FAIL_TO_PASS']))
    S['p2p_gt30_instances']=sum(1 for r in F if len(r['PASS_TO_PASS'])>30)
    # gold
    gp=[r['gold_patch'] for r in F]
    S['gold_patch_nonempty']=sum(1 for g in gp if g.strip())
    S['gold_patch_starts_diff_git']=sum(1 for g in gp if g.startswith('diff --git'))
    S['gold_patch_starts_---']=sum(1 for g in gp if g.startswith('--- '))
    S['gold_patch_malformed']=[r['instance_id'] for r in F if not (r['gold_patch'].startswith('diff --git') or r['gold_patch'].startswith('--- '))]
    S['gold_patch_len_dist']=dist([len(g) for g in gp])
    S['gold_patch_has_index_line']=sum(1 for g in gp if re.search(r'^index [0-9a-f]+\.\.[0-9a-f]+',g,re.M))
    gt=[]; gsub=[]; nfiles=[]; mism=[]
    for r in F:
        paths=touched_paths(r['gold_patch']); nfiles.append(len(paths))
        tf={x.split('::')[0] for x in r['FAIL_TO_PASS']+r['PASS_TO_PASS'][:30]}
        hit=[p for p in paths if p in tf or _TESTISH.search(p)]
        if hit: gt.append({'instance_id':r['instance_id'],'paths':hit})
        if L[r['instance_id']]['num_patched_files']!=len(paths): mism.append((r['instance_id'],L[r['instance_id']]['num_patched_files'],len(paths)))
        try: bf=json.loads(r['buggy_files'])
        except Exception: bf=None
        if isinstance(bf,dict) and not set(paths)<=set(bf): gsub.append({'instance_id':r['instance_id'],'gold_paths':paths,'buggy_keys':list(bf)})
    S['gold_touches_test_or_config_files']={'n':len(gt),'examples':gt[:10]}
    S['gold_files_touched_dist']=dist(nfiles)
    S['num_patched_files_mismatch_vs_gold']={'n':len(mism),'examples':mism[:10]}
    S['gold_paths_not_subset_of_buggy_files']={'n':len(gsub),'examples':gsub[:5]}
    S['patch_direction_values']=vc([r['patch_direction'] for r in F])
    S['exec_backend_values']=vc([r['exec_backend'] for r in F])
    # buggy files
    bfs=[]; bad=[]
    for r in F:
        try:
            bf=json.loads(r['buggy_files']); bfs.append(bf)
            if not isinstance(bf,dict) or not bf: bad.append(r['instance_id'])
        except Exception: bad.append(r['instance_id']); bfs.append(None)
    S['buggy_files_parse_ok']=sum(1 for b in bfs if isinstance(b,dict))
    S['buggy_files_empty_or_bad']=bad
    S['buggy_files_nfiles_dist']=dist([len(b) for b in bfs if isinstance(b,dict)])
    S['buggy_files_total_chars_dist']=dist([sum(len(v) for v in b.values()) for b in bfs if isinstance(b,dict)])
    S['buggy_files_raw_len_dist']=dist([len(r['buggy_files']) for r in F])
    # instance id structure / branch identification
    kinds=[]; commit_ok=0; repo_ok=0; unparsed=[]
    for r in F:
        m=IID_RE.match(r['instance_id'])
        if not m: unparsed.append(r['instance_id']); continue
        kinds.append(m.group('kind'))
        if r['image_name'].endswith('.'+m.group('commit')): commit_ok+=1
        if r['repo'].lower()=='swesmith/'+(m.group('owner')+'__'+m.group('repo')+'.'+m.group('commit')).lower(): repo_ok+=1
    S['instance_id_parse']={'parsed':len(F)-len(unparsed),'unparsed_examples':unparsed[:10],'bug_kind_values':vc(kinds),'commit_prefix_matches_image_tag':commit_ok,'repo_field_matches_owner_repo_commit':repo_ok}
    S['instance_id_len_dist']=dist([len(r['instance_id']) for r in F])
    # problem statement
    ps=[r['problem_statement'] for r in F]; ko=[r['problem_statement_ko'] for r in F]
    S['problem_statement_len_dist']=dist([len(x) for x in ps]); S['problem_statement_ko_len_dist']=dist([len(x) for x in ko])
    S['problem_statement_empty']=sum(1 for x in ps if not x.strip())
    S['problem_statement_ko_identical_to_en']=sum(1 for a,b in zip(ps,ko) if a==b)
    S['problem_statement_ko_contains_hangul']=sum(1 for x in ko if re.search(r'[가-힣]',x))
    S['problem_statement_exact_dupes']={k:v for k,v in collections.Counter(ps).items() if v>1}.__len__()
    # learnable fields
    Ls=learn[t]
    S['p_hat_values']=vc([r['p_hat'] for r in Ls]); S['p_hat_dist']=dist([r['p_hat'] for r in Ls])
    S['p_hat_source_values']=vc([r['p_hat_source'] for r in Ls])
    S['difficulty_0_10_values']=vc([r['difficulty_0_10'] for r in Ls]); S['difficulty_band_values']=vc([r['difficulty_band'] for r in Ls])
    S['difficulty_model_values']=vc([r['difficulty_model'] for r in Ls]); S['judge_difficulty_values']=vc([r['judge_difficulty'] for r in Ls])
    S['judge_difficulty_x_p_hat']=vc([f"{r['judge_difficulty']}->p_hat={r['p_hat']},d={r['difficulty_0_10']}" for r in Ls])
    S['judge_verdict_values']=vc([r['judge_verdict'] for r in Ls])
    qk=sorted({k for r in Ls for k in r['quality']})
    S['quality_fields']=qk
    S['quality_value_dist']={k:vc([json.dumps(r['quality'].get(k)) for r in Ls]) for k in qk}
    S['judge_verdict_x_artifact_leak']=vc([f"{r['judge_verdict']}|leak={r['quality'].get('artifact_leak')}" for r in Ls])
    S['judge_verdict_x_issue_quality']=vc([f"{r['judge_verdict']}|iq={r['quality'].get('issue_quality')}|sm={r['quality'].get('scope_match')}|ta={r['quality'].get('test_alignment')}" for r in Ls])
    S['judge_verdict_x_band']=vc([f"{r['judge_verdict']}|{r['difficulty_band']}" for r in Ls])
    S['drop_instance_ids']=[r['instance_id'] for r in Ls if r['judge_verdict']=='drop']
    S['context_bucket_values']=vc([r['context_bucket'] for r in Ls]); S['num_patched_files_values']=vc([r['num_patched_files'] for r in Ls])
    for k in ('track','data_source','ability','context_mode','subset','exec_backend'): S[f'{k}_values']=vc([r[k] for r in Ls])
    return S
R['section2_swe']=swe_section('t15_code_swesmith')

# ---------- Section 4: R2E ----------
S4=swe_section('t15_code_r2e')
S4['test_patch_in_full_keys']=any('test_patch' in r for r in full['t15_code_r2e'])
S4['test_patch_in_learnable_keys']=any('test_patch' in r for r in learn['t15_code_r2e'])
S4['images_in_r2e_txt']=sum(1 for r in full['t15_code_r2e'] if r['image_name'] in img_r2e)
S4['images_in_manifest_json']=sum(1 for r in full['t15_code_r2e'] if r['image_name'] in manifest_imgs)
S4['f2p_id_format_examples']=full['t15_code_r2e'][0]['FAIL_TO_PASS'][:3]
R['section4_r2e']=S4

# ---------- Section 3: unittest ----------
def ut_section(t):
    F=full[t]; L={r['instance_id']:r for r in learn[t]}; Ls=learn[t]
    S={'rows':len(F)}
    S['harness_values']=vc([r['harness'] for r in F])
    S['learnable_harness_matches_full']=all(L[r['instance_id']]['harness']==r['harness'] for r in F)
    nt=[len(r['tests']) for r in F]
    S['n_tests_dist']=dist(nt); S['n_tests_dist_by_harness']={h:dist([len(r['tests']) for r in F if r['harness']==h]) for h in S['harness_values']}
    S['zero_tests_instances']=[r['instance_id'] for r in F if len(r['tests'])==0]
    S['learnable_n_tests_mismatch']=[r['instance_id'] for r in F if L[r['instance_id']]['n_tests']!=len(r['tests'])]
    S['test_item_keys_by_harness']={h:vc([json.dumps(sorted(x.keys())) for r in F if r['harness']==h for x in r['tests']]) for h in S['harness_values']}
    S['test_kind_values']=vc([x.get('kind') for r in F for x in r['tests']])
    S['empty_assertion_items']=sum(1 for r in F for x in r['tests'] if r['harness']!='stdio' and not (x.get('assertion') or '').strip())
    # pytest harness: number of def test_ functions per instance
    pyt=[sum(len(re.findall(r'^\s*def\s+test\w*\s*\(',x.get('assertion') or '',re.M)) for x in r['tests']) for r in F if r['harness']=='pytest']
    S['pytest_test_functions_per_instance_dist']=dist(pyt)
    S['pytest_instances_with_zero_test_functions']=sum(1 for v in pyt if v==0)
    S['pytest_tests_importing_solution']=sum(1 for r in F if r['harness']=='pytest' and any('from solution import' in (x.get('assertion') or '') or 'import solution' in (x.get('assertion') or '') for x in r['tests']))
    S['reference_solution_nonempty']=sum(1 for r in F if (r['reference_solution'] or '').strip())
    S['has_reference_true']=sum(1 for r in Ls if r['has_reference'])
    S['has_reference_vs_nonempty_mismatch']=[r['instance_id'] for r in F if bool((r['reference_solution'] or '').strip())!=bool(L[r['instance_id']]['has_reference'])]
    S['reference_solution_len_dist']=dist([len(r['reference_solution'] or '') for r in F if (r['reference_solution'] or '').strip()])
    S['entry_point_nonnull_by_harness']={h:sum(1 for r in F if r['harness']==h and r['entry_point']) for h in S['harness_values']}
    S['entry_point_examples']=[r['entry_point'] for r in F if r['entry_point']][:5]
    S['instruction_nonempty']=sum(1 for r in F if (r['instruction'] or '').strip())
    S['statement_len_dist']=dist([len(r['statement']) for r in F]); S['statement_empty']=sum(1 for r in F if not r['statement'].strip())
    S['statement_contains_hangul']=sum(1 for r in F if re.search(r'[가-힣]',r['statement']))
    S['time_limit_s_values']=vc([r['time_limit_s'] for r in F]); S['memory_mb_values']=vc([r['memory_mb'] for r in F])
    S['instance_id_prefix_values']=vc([r['instance_id'].split('-')[0] if '-' in r['instance_id'] else r['instance_id'].split('_')[0] for r in F])
    for k in ('lang','upstream','p_hat_source','difficulty_band','difficulty_model','exec_backend','track','data_source','ability','context_mode','subset'):
        S[f'{k}_values']=vc([r[k] for r in Ls])
    S['upstream_x_harness']=vc([f"{r['upstream']}|{r['harness']}" for r in Ls])
    S['upstream_x_lang']=vc([f"{r['upstream']}|{r['lang']}" for r in Ls])
    ph=[r['p_hat'] for r in Ls if r['p_hat'] is not None]
    S['p_hat_nonnull']=len(ph); S['p_hat_dist']=dist(ph); S['p_hat_values']=vc(ph)
    S['difficulty_0_10_values']=vc([r['difficulty_0_10'] for r in Ls])
    S['p_hat_out_of_band_0.2_0.8']=sum(1 for p in ph if not (0.2<=p<=0.8))
    S['band_x_p_hat_consistency_violations']=[r['instance_id'] for r in Ls if r['p_hat'] is not None and (r['difficulty_band']=='learnable')!=(0.2<=r['p_hat']<=0.8)]
    return S
R['section3_unittest_core']=ut_section('t15_code_unittest')
R['section3_unittest_ext']=ut_section('t15_code_unittest_ext')

R['section6_near_duplicates']={'see':'_dupes_part.json'}
# ---------- Section 7 (index side) ----------
S7={}
S7['swe_gold_malformed']=R['section2_swe']['gold_patch_malformed']
S7['r2e_gold_malformed']=S4['gold_patch_malformed']
S7['swe_empty_problem_statement']=R['section2_swe']['problem_statement_empty']
S7['unittest_core_empty_statement']=R['section3_unittest_core']['statement_empty']; S7['unittest_ext_empty_statement']=R['section3_unittest_ext']['statement_empty']
S7['unittest_core_zero_tests']=R['section3_unittest_core']['zero_tests_instances']; S7['unittest_ext_zero_tests']=R['section3_unittest_ext']['zero_tests_instances']
S7['swe_zero_f2p']=R['section2_swe']['zero_f2p_instances']; S7['r2e_zero_f2p']=S4['zero_f2p_instances']
S7['swe_images_not_in_manifest']=R['section2_swe']['images_not_in_manifest']; S7['r2e_images_not_in_manifest_count']=len(full['t15_code_r2e'])-S4['images_in_manifest_json']
S7['swe_gold_touches_tests_n']=R['section2_swe']['gold_touches_test_or_config_files']['n']
S7['index_text_len_gt_105k_chars']={t:sum(1 for r in full[t] if len(r.get('problem_statement') or r.get('statement') or '')>105000) for t in TAGS}
S7['swe_buggy_files_gt_105k_chars']=sum(1 for r in full['t15_code_swesmith'] if len(r['buggy_files'])>105000)
R['section7_index_side']=S7

# ---------- per-instance base rows ----------
base={}
for t in TAGS:
    L={r['instance_id']:r for r in learn[t]}
    rows=[]
    for r in full[t]:
        l=L[r['instance_id']]
        if t in ('t15_code_swesmith','t15_code_r2e'):
            try: hb=bool(json.loads(r['buggy_files']))
            except Exception: hb=False
            rows.append({'instance_id':r['instance_id'],'repo_or_image':r['repo'],'data_source':l['data_source'],'n_f2p':len(r['FAIL_TO_PASS']),'n_p2p':len(r['PASS_TO_PASS']),
                'p_hat':l['p_hat'],'difficulty_band':l['difficulty_band'],'subset':l['subset'],'has_gold_patch':bool(r['gold_patch'].strip()),'has_buggy_files':hb,
                'index_text_len_chars':len(r['problem_statement']),'harness_kind':r['exec_backend'],'language':'','judge_verdict':l.get('judge_verdict')})
        else:
            rows.append({'instance_id':r['instance_id'],'repo_or_image':'','data_source':l['data_source'],'n_f2p':len(r['tests']),'n_p2p':0,
                'p_hat':l['p_hat'],'difficulty_band':l['difficulty_band'],'subset':l['subset'],'has_gold_patch':bool((r['reference_solution'] or '').strip()),'has_buggy_files':False,
                'index_text_len_chars':len(r['statement']),'harness_kind':r['harness'],'language':l['lang'],'judge_verdict':None})
    base[t]=rows
json.dump(base,open(f'{OUT}/_base_rows.json','w'))
json.dump(R,open(f'{OUT}/_index_part.json','w'),indent=1,ensure_ascii=False,default=str)
print('index audit done')

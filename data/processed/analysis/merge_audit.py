#!/usr/bin/env python3
import json, csv, os, datetime, collections
A='/scratch/r919a03/code_verl_test/data/processed/analysis'
I=json.load(open(f'{A}/_index_part.json')); Pq=json.load(open(f'{A}/_parquet_part.json')); Du=json.load(open(f'{A}/_dupes_part.json'))
Ps=json.load(open(f'{A}/_probe_swe.json')); Pu=json.load(open(f'{A}/_probe_ut.json')); Pf=json.load(open(f'{A}/_probe_followup.json')); Pe=json.load(open(f'{A}/_probe_extra.json'))
plen=json.load(open(f'{A}/_prompt_len.json')); base=json.load(open(f'{A}/_base_rows.json'))
SRC='/scratch/r919a03/code_verl_test/sources/rl_code_v1'
out={'meta':{'generated_at':datetime.datetime.now().isoformat(timespec='seconds'),'source_dir':f'{SRC}/data','python':'/scratch/r919a03/.conda/envs/py312/bin/python (pandas 3.0.0, pyarrow 23.0.0)',
   'scripts':[f'{A}/audit_index.py',f'{A}/audit_parquet.py',f'{A}/audit_dupes.py',f'{A}/probe_swe_prompt.py',f'{A}/probe_ut_names.py',f'{A}/probe_followup.py',f'{A}/probe_extra.py',f'{A}/merge_audit.py'],
   'token_estimate':'approx_tokens = chars/3.5 (no tokenizer run unless noted in tokenizer_calibration)'}}
out['section1_index_files']=I['section1_index_files']
s2=I['section2_swe']; s2['prompt_probe']=Ps; s2['gold_vs_judge_test_gate']=Pf['swe_gold_would_be_rejected_by_touches_tests']
s2['f2p_ids_in_non_test_files_by_repo']=Pf['swe_f2p_ids_whose_file_is_not_testish_by_repo']; s2['f2p_nontest_examples']=Pf['swe_f2p_nontestish_examples']
s2['inflect_check']={k:Pe[k] for k in ('inflect_n','inflect_gold_touches_init','inflect_f2p_file_is_init','inflect_f2p_files')}
s2['judge_verdict_crosstabs']={k:Pf[k] for k in ('verdict_x_scope_match','verdict_x_test_alignment','verdict_x_issue_quality','verdict_x_judge_difficulty','verdict_x_bugkind','fix_verdict_rows')}
s2['ko_translation_null_iff_ps_ko_identical']=Pf['ko_translation_null_iff_ps_ko_identical']
s2['snapshot_files_included_dist']=Pe['swe_snapshot_files_included_dist']; s2['snapshot_files_by_variant']=Pe['swe_snapshot_files_by_variant']
s2['issue_text_mentions_gold_file_path']=Pe['swe_issue_mentions_gold_path']
s2['instances_per_repo']=Pe['swe_instances_per_repo_full']
s2['base_commit_explanation']=('No base_commit/branch/sha field exists in full or learnable records (base_commit_like_fields_present=[]). '
  'The buggy checkout is identified by git branch origin/<instance_id> inside the per-repo image (reward/swe_exec_v2.py:104), with test files restored from `main` (line 105); '
  'the repo snapshot commit is the 8-hex suffix embedded in instance_id / image tag / repo field (748/751 parse; the 3 Knio__dominate.9082227e.pr_* ids lack the <kind>__<hash> suffix but still carry the commit).')
out['section2_swe']=s2
out['section3_unittest']={'core':I['section3_unittest_core'],'ext':I['section3_unittest_ext'],'n_tests_semantics':{k:v for k,v in Pf.items() if k.startswith('n_tests_semantics')},
   'function_name_in_statement_probe':Pu,'prompt_vs_statement':Pf['unittest_prompt_vs_statement'],'p_hat_source_x_p_hat_core':Pe['p_hat_source_x_p_hat_core'],
   'pytest_example':{'instance_id':'kodcode-a154fafde017','test_file_head':Pf['pytest_example_test_file'],'entry_point':Pf['pytest_example_entry_point']},
   'function_example':{'instance_id':'opencodeinstruct-8fe7c2210ce5','statement':Pf['function_example_statement'],'tests':Pf['function_example_tests']}}
s4=I['section4_r2e']; s4['r2e_total_f2p_ids']=Pe['r2e_total_f2p_ids']; s4['r2e_f2p_ids_with_double_colon']=Pe['r2e_f2p_ids_with_::']
s4['exclusion_reasons']=['docs/ENVIRONMENT.md:119-124 and README.md:127-128: exec harness kind=gym requires test_patch (reward/swe_exec_v2.py:88,112) and the records have no test_patch field (test_patch_in_full_keys=False)',
  'FAIL_TO_PASS ids are unittest-style Class.test (0/6842 contain "::"); reward/swe_exec_v2.py:33-44 _clean_ids drops ids without "::" -> no_valid_f2p -> err_kind infra_data (t15_code_judge.py:147)',
  'PASS_TO_PASS empty for all 120; images (namanjain12/aiohttp_final:*) are not in images_manifest.json (0/120) and no pull script covers them',
  'not present in any parquet (index_rows_not_in_parquet = {t15_code_r2e:120})']
out['section4_r2e']=s4
s5=Pq; s5['unittest_image_name_values']=Pf['extra_info.image_name_values_unittest']
s5['rubrics']={'reward_true_count':Pf['rubric_tag_reward_true_count'],'judge_true_count':Pf['rubric_tag_judge_true_count'],'verify_values':Pf['rubric_tags_verify_values'],'unittest_top_criteria':Pf['rubric_criteria_unittest_top'],'swe_top_criteria':Pf['rubric_criteria_swe_top']}
s5['identical_prompt_groups_detail']=Pe['identical_prompt_groups_detail']; s5['val_rows_with_user_msg_in_train']=Pe['val_rows_with_user_msg_in_train']
s5['swe_rows_est_tokens_gt_32768_by_variant']=Pf['swe_rows_est_tokens_gt_32768_by_variant']; s5['swe_rows_est_tokens_le_32768_by_variant']=Pf['swe_rows_est_tokens_le_32768_by_variant']
s5['join_logic']='reward looks up extra_info.instance_id, falling back to reward_model.ground_truth (reward/t15_code_reward.py:108); both equal extra_info.index in 100% of rows; unittest rows join to t15_code_unittest*.full.jsonl (t15_code_reward.py:63), SWE rows to t15_code_swesmith*.learnable+full.jsonl (t15_code_judge.py:65-74)'
s5['output_format_instructions']={'unittest_system':'You are an expert programmer. Solve the problem and reply with a single ```python code block containing the complete solution.',
  'swe_system':'<solution> ### path <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE </solution> edit blocks (see example_prompt_swe.txt)',
  'swe_user_tail_en':'Fix the issue by editing the repository files. Reply with a single unified diff inside a ```diff code block. Use paths relative to the repository root (--- a/<path> / +++ b/<path>) ... do not modify test files.',
  'swe_user_tail_ko':'위 이슈를 저장소 파일을 수정해 해결하세요. 답은 ```diff 코드블록 안에 unified diff 하나로 작성합니다. ...',
  'contradiction':'system asks SEARCH/REPLACE in 751/751 SWE rows while the user message tail asks for a unified ```diff block in 751/751 rows; reward/patch_extract.py accepts both (6-stage cascade)'}
out['section5_parquet']=s5
out['section6_near_duplicates']=Du
s7=I['section7_index_side']
s7['parquet_empty_prompts']={n:v['prompt_empty_rows'] for n,v in Pq['per_file'].items()}
s7['parquet_prompt_gt_30k_tokens_est']={n:v['prompt_gt_30k_tokens_est'] for n,v in Pq['per_file'].items()}
s7['parquet_prompt_gt_32768_tokens_est']={n:v['prompt_gt_32768_tokens_est'] for n,v in Pq['per_file'].items()}
s7['swe_gold_rejected_by_judge_test_gate']=Pf['swe_gold_would_be_rejected_by_touches_tests']['total']
s7['swe_p2p_gt30_truncated_by_judge']=I['section2_swe']['p2p_gt30_instances']
s7['unittest_rows_entry_point_absent_from_prompt']=Pf['unittest_prompt_vs_statement'].get('entry_point_NOT_in_user_msg')
s7['identical_prompt_groups_across_parquet']=len(Pe['identical_prompt_groups_detail'])
out['section7_suspicious_rows']=s7
json.dump(out,open(f'{A}/dataset_audit.json','w'),indent=1,ensure_ascii=False,default=str)
# ---- CSVs ----
cols=['instance_id','repo_or_image','data_source','n_f2p','n_p2p','p_hat','difficulty_band','subset','has_gold_patch','has_buggy_files','prompt_len_chars','harness_kind','language']
for tag,rows in base.items():
    with open(f'{A}/instances_{tag}.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in rows:
            p=plen.get(r['instance_id'])
            d={c:r.get(c,'') for c in cols}
            if p:
                d['prompt_len_chars']=p[0]
                if not d['language']: d['language']=p[3] or ''
                if tag.startswith('t15_code_unittest'): d['repo_or_image']=p[4] or ''
            else:
                d['prompt_len_chars']=''
                if tag=='t15_code_r2e': d['repo_or_image']=next(x for x in [r['repo_or_image']])  # keep repo
            d['p_hat']='' if d['p_hat'] is None else d['p_hat']
            w.writerow(d)
    print(tag,len(rows))
print('written',f'{A}/dataset_audit.json')

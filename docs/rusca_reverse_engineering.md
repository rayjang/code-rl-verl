# RUSCA / RL-algorithm reverse engineering

Status legend: **CONFIRMED** = read in local source; **PUBLIC** = taken from the published paper/verl
source and consistent with local hints; **UNKNOWN** = not recoverable because the original bundles
(`t15_rl_bundle.tar.gz`, `t01_math_kit.tar.gz`) are missing from this machine (see
`docs/source_inventory.md` §1). Full evidence with line numbers:
`data/processed/analysis/reports/algo_evidence.md` (+ `.verify.md`).

## 1. What RUSCA is in this project

| Item | Status | Evidence / decision |
|---|---|---|
| Definition | CONFIRMED | RuscaRL = *Rubric-Scaffolded RL* (Zhou et al., arXiv 2508.16949). Locally: rubric criteria are injected into the **rollout prompt only**; the reward is rule-based end-to-end; advantages are untouched (`sources/rl_code_v1/docs/MATH_REWARD.md:114-118`, `sources/t01_math_reward_rubric/t01_math_reward.py:11-12`). |
| Rubric schema | CONFIRMED | `{index, criteria[{criterion, points, tags{judge, inject, scaffold_rank, scaffold_kind, verify, reward}}]}` (`t01_math_reward_rubric/README.md:76-81`). The code profile in `rl_code_v1` parquet `extra_info.rubrics` uses the same schema: 5–6 criteria per row, `scaffold_kind=hint`, `verify ∈ {failing_test, impl_hint, approach, edge_cases, complexity, tests_pass, output_format, no_extra}`, ranks 0–4, all `inject=true`, `reward=false`, `judge=false` (measured over 6,516 train rows). |
| Temporal decay | CONFIRMED (table) / PUBLIC (formula) | Local validation table (`MATH_REWARD.md:129-134`, code profile, 50 steps): lam(0)=1.0, lam(10)=0.5, lam(12)=0.0067, lam(50)=0 → "scaffold fully removed at 24 %". This matches the paper's `λ_step(t)=1/(1+e^{α(t−t0)})` with `t0=0.2, α=125` (t = step/total). Implemented in `verifier/rusca.py::lam` (center_frac=0.2, steepness scaled to total_steps) and unit-tested against the three points. |
| Group-level differentiation | PUBLIC + local hint | Paper: `λ_i=(G−i)/(G−1)` linear intra-group decay. Locally only the count row "6 6 6 5 6 5" at lam=1 is visible → the exact intra-group rule is **UNKNOWN**; `verifier/rusca.py::n_inject` uses `round(lam·n_criteria·(1±jitter))` with a deterministic per-variant jitter (config `group_jitter`). |
| Drop order | CONFIRMED | Criteria are removed by `scaffold_rank` (math: answer_hint/step_hint(0) → strategy/subgoal(1) → key_condition(2) → show_work/self_check(3) → answer_format/no_skip(4), `MATH_REWARD.md:121-122`). Code profile ranks observed in data: failing_test/impl_hint(0) → approach(1) → edge_cases/complexity(2) → tests_pass(3) → output_format/no_extra(4). `select_criteria` keeps the k lowest-rank criteria. |
| Injection text | UNKNOWN (template) | Paper template (App. D.2) = "consider the following evaluation criteria … do not mention them". Local template not available. `verifier/rusca.py::inject` appends "Follow these guidelines:" + bullet list to the last user turn (ko/en). |
| Agent loop | CONFIRMED (interface) / UNKNOWN (code) | Launched as `rollout.agent.default_agent_loop=rusca_scaffold_agent`, `agent_loop_config_path=rusca_agent_loop.yaml`, `+rollout.custom.rusca.{enable,profile,total_steps}`, **`calculate_log_probs=False`** because rollout log-probs are scaffold-conditioned (`t01_math_reward_rubric/README.md:105-118`). The paper (App. C.6) trains with the no-scaffold ratio, i.e. old log-probs are recomputed on the un-scaffolded prompt. Re-implemented in `rl/rusca_agent_loop.py`: returns **un-scaffolded prompt ids + response ids**, so actor/ref log-probs are computed on the training prompt. |
| Rule-only transition | CONFIRMED | Scaffold count → 0 by ~24 % of steps; from then on rollouts are unscaffolded and the reward was always rule-based. Our `RewardConfig.rusca_stage ∈ {scaffold, transition, rule}` additionally allows a rubric weight during the first two stages (ablation "RUSCA reward"), default 0 to match the local semantics. |
| Measured effect (math) | CONFIRMED | Category rubrics: 10.9 % vs 10.7 % baseline exploration opening (z=+0.06, no effect); per-problem hints: 16.4 % (4B) / 32.3 % (14B) (`t01_math_reward_rubric/README.md:124-136`). |

## 2. Reward flow (multi-task)

```
verl reward loop worker
  └─ custom_reward_function = t15_reward_dispatch.compute_score   (baseline)
       ├─ data_source t15_repo_patch / t15_unittest_impl → t15_code_reward.compute_score_dict
       │     ├─ SWE: t15_code_judge.judge → patch_extract → touches_tests gate → swe_exec_v2.run_one (singularity)
       │     └─ UT : unittest_exec.judge_unittest (function / pytest / stdio, canary)
       │     └─ ladder score_of(state, f2p_frac, p2p_frac) → overlong shaping → 24-key dict
       ├─ t01_* → T01_SCORER (math, `t01_math_reward.py`)            [slot lives in missing t15_reward_multi.py]
       └─ t03_* / other t15_* → t15_reward_multi.py                  [MISSING: /scratch/r919a13/data/T15_longctx]
  key-union template built at import; None/NaN coerced (dispatch_none_coerced)
```
Contract (CONFIRMED, `t15_code_reward.py:8-19`, `REWARD_DESIGN.md:220-235`): every row returns the same key
set; float/str values; deterministic; keys `score`, `answer_match` (pure 0/1 = resolved), `score_before_overlong`.
Our `verifier/verl_reward.py` keeps this contract (33 keys, template-locked after the first call).

## 3. Algorithm components

| Component | Status | Semantics used in this project |
|---|---|---|
| GRPO advantage | CONFIRMED | `algorithm.adv_estimator=grpo` (`verl_example.sh:33`); group size n≥8 recommended (dead-group table `HANDOFF.md:142-150`). |
| std on | UNKNOWN locally / PUBLIC | verl `algorithm.norm_adv_by_std_in_grpo=True` (default). Used as such. |
| GSPO loss | UNKNOWN locally / PUBLIC | verl `actor.policy_loss.loss_mode=gspo` (`core_algos.py:1545`): sequence-level ratio `s_i=exp(mean_t log π/π_old)`, clip (paper: 3e-4 / 4e-4). **verl's GSPO hard-codes `seq-mean-token-mean` aggregation**, so "GSPO + token-mean" is not expressible with stock verl; `rl/algos.py` registers `gspo_tokenmean` (same ratio, `loss_agg_mode=token-mean`). |
| token mean | UNKNOWN locally / PUBLIC | `actor.loss_agg_mode=token-mean` (verl default). Applied to vanilla / gspo_tokenmean losses. |
| DDCA | PUBLIC (paper identified) | *Think Dense, Not Long: Dynamic Decoupled Conditional Advantage* (arXiv 2602.02099): `A_i = A_i^acc − β·A_i^len`; correct cluster `C={r^acc=1}`; `r_i^len = sigmoid(z_i)` with z-score of length inside C; `A_i^len=(n/N)·(r_i^len − mean_{j∈C, j≠i} r_j^len)` for i∈C, 0 otherwise. Local scorers require `answer_match` to be pure correctness "because DDCA builds the correct cluster from it" (`t15_code_reward.py:14-18`). Implemented as advantage estimator `grpo_ddca` in `rl/algos.py` (GRPO std-normalised accuracy advantage − β·DDCA length advantage; β config). **Local β and whether the local variant used RLOO or std baselines are UNKNOWN.** |
| Hierarchical sampler | UNKNOWN locally | Only the description "task-level sampling then variant-level sampling" exists. verl 0.9 has no sampler hook (`data.sampler.class_path` removed after v0.7). Implemented as `rl/hier_sampler.py` (task=`data_source`, variant=`variant_type`, weights, without-replacement per epoch, resumable) and injected via the patched TaskRunner in `rl/main_ppo_code.py`. |
| Multi-task reward | CONFIRMED (dispatcher) | Routing by `data_source` prefix + key-union template (`t15_reward_dispatch.py`). Ours: `verifier/verl_reward.py` routes SWE vs unit-test tracks, same-key guarantee. Math track not in scope (kit missing). |
| Overlong shaping | CONFIRMED | `pen = −min(1, (len − (max_len − buffer))/buffer)·penalty` (all three scorers); default off. Ported to `verifier/reward.py::overlong`. |

## 4. Input / output schemas

* Parquet row: `data_source`, `prompt` (chat list), `ability`, `reward_model{ground_truth=instance_id, style}`,
  `extra_info{index, split, task_tag, rubric_profile, rubrics[], variant_type, instance_id, exec_backend, harness,
  p_hat, p_hat_source, difficulty_0_10, difficulty_band, n_f2p, n_p2p, n_tests, context_bucket, lang, image_name, repo}`.
* Reward dict (ours): see `verifier/schemas.py::RewardComponents` — every component is logged
  (extraction/format/apply/F2P/P2P/rubric/rule/timeout/length/final + infra_excluded + err_kind + versions).

## 5. Conflicts between local materials (reported, not hidden)

1. The SWE system prompt asks for `<solution>` SEARCH/REPLACE blocks while the user turn asks for a
   ```` ```diff ```` unified diff (both in every SWE prompt). The verifier therefore accepts both formats
   (S/R converted only with exact unique matches).
2. `README.md` claims all gold patches pass their tests; the SWE track was never executed by the authors
   (`HANDOFF.md:196-199`). Our validation found 47 instances whose bug lives in a restored test/doctest file,
   32 whose gold applies only with `--ignore-whitespace`, and repo-level test-environment failures
   (see `docs/dataset_quality_report.md`).
3. The baseline dispatcher references a modified verl (`ray_trainer.py:315` DDCA, `+rollout.custom.rusca`);
   stock verl 0.9.0 has neither, hence the re-implementations above.

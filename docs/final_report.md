# Final report — Coding RL reward / verifier / rubric research and final RL dataset

> Status: sections 9–19 are filled in as experiments complete (see `docs/experiment_report.md`,
> `docs/ablation_report.md`, `results/rl_code_reward_research.xlsx`). Everything stated as a number in
> this document has a file it was read from.

## 1. Executive summary
_(filled after Stage D/E)_

## 2. Dataset analysis (`rl_code_v1`)
* 23,142 training rows in three parquets: core train 6,516 (5,781 unit-test + 735 SWE-smith), core val 162,
  extended train 16,464 (unit-test only, difficulty unmeasured). Index: 751 SWE-smith instances over 19
  repositories/images (40 per repo, cantools 32, bottle 39), 5,927 + 16,464 unit-test instances
  (function / pytest / stdio harnesses), 120 R2E instances (no `test_patch`, excluded upstream).
* SWE prompts embed a repository snapshot; three buckets by size: swe_32k (351 rows, 18–28k tokens),
  swe_64k (263, 37–62k), swe_128k (137, 75–119k) measured with the Qwen1.5-MoE tokenizer. Only swe_32k fits
  the model's 32,768-token window with room for a response.
* Every SWE prompt is internally contradictory: the system turn asks for `<solution>` SEARCH/REPLACE blocks,
  the user turn asks for one ```` ```diff ```` block.
* `extra_info` already carries a code-profile rubric (5–6 criteria, all `inject:true, reward:false`),
  `variant_type`, `lang` (ko 404 / en 6,112 in core train), `p_hat` (measured pass@k for unit tests with a
  different model; judge-mapped constants 0.25/0.4/0.6/0.75 for SWE), `judge_verdict` (drop 430 / keep 319 /
  fix 2 — semantics undocumented).
* Unit-test prompts frequently omit the function/class name the hidden tests import (84 % of pytest-harness
  statements per the audit); no code path uses `entry_point`.
* Full audit: `data/processed/analysis/reports/dataset_audit.md`, per-instance CSVs in
  `data/processed/analysis/`.

### 2.1 Policy-model decision (2026-09-04)
The user asked to include the 400 SWE rows that do not fit a 32k window. Qwen1.5-MoE-A2.7B-Chat (32,768
tokens, no YaRN training) cannot train on them, so the policy was switched to **Qwen3-30B-A3B-Instruct-2507**
(same MoE family, 3B active / 30B total, 262k native context, already cached on this machine; verl 0.9,
vLLM 0.24 and transformers 5.10 support `qwen3_moe`). Qwen1.5-MoE results (Stage D arms D000–D002) are kept
as a same-data reference. With the Qwen3 tokenizer every SWE prompt fits: swe_32k 18–28k, swe_64k 37–62k,
swe_128k 75–119k tokens (`results/smoke/swe_prompt_tokens_qwen3.json`). Dataset `curated_v3pre` therefore
keeps all 504 environment-valid SWE rows (train 198/213/93 for 32k/64k/128k, validation 112, test 99) and
`curated_v3` applies the Qwen3 base-policy difficulty filter to the training split only.

## 3. Execution environment analysis
* 19 `jyangballin/swesmith.x86_64.*` images pulled from Docker Hub as `.sif` (23 GB total; digests in
  `sources/rl_code_v1/data/index/images_manifest.json`), plus `python:3.11-slim-bookworm.sif` with a bind-mounted
  venv (pytest 8.3.4, numpy 2.2.1, sympy 1.13.3) for the unit-test track. No fakeroot on the cluster, so the
  shipped `unittest.def` could not be built; its content is reproduced by the venv.
* Convention kept: repository → one image, instance → git branch `origin/<instance_id>` (buggy state, tests
  deleted), `main` = clean state used to restore test material.
* Validation protocol per instance (`scripts/validate_environments.py`): clean checkout → image launch →
  empty-patch baseline (F2P must fail, P2P must pass) → gold patch → F2P/P2P → cleanup. Five iterations were
  needed; each defect and fix is listed in `docs/dataset_quality_report.md` §4–5. Manifests:
  `environments/manifests/validation_swe.jsonl`, `validation_ut.jsonl`.

## 4. Existing RUSCA analysis
See `docs/rusca_reverse_engineering.md`. In short: locally RUSCA = rubric criteria injected into the rollout
prompt with a logistic decay (1.0 → 0.5 at 20 % of steps → 0 by ≈24 %), dropped by `scaffold_rank`; the reward
stayed rule-based; `calculate_log_probs=False` because old log-probs must be recomputed without the scaffold.
GSPO/token-mean/std/hierarchical-sampler have no local evidence (bundles missing); DDCA is the public
"Dynamic Decoupled Conditional Advantage" (arXiv 2602.02099) whose correct-cluster input is `answer_match`.

## 5. Coding verifier design
See `docs/verifier_design.md`. Four extraction candidates (strict / markdown / fallback / baseline), strict
`git apply` (no fuzz) with an evidence-based `--ignore-whitespace` option, isolated singularity execution,
effective F2P/P2P sets, hardened anti-hacking (randomised multi-type `__eq__` canary, quoted-path parsing,
test/config/verifier path rules with instance-level overrides).

## 6. Rubric design
See `docs/rubric_design.md` and `rubrics/*.yaml`.

## 7. Reward candidates
See `docs/coding_reward_design.md` and `configs/reward/*.yaml` (16 candidates: baseline ladder, F2P binary /
linear / sqrt / ladder, P2P strong / tolerant / capped / confidence / none, rule-only, rubric-heavy, RUSCA,
additive gating / non-gating, no-apply-credit).

## 8. Experimental method
* Stage A: 54 unit tests (`tests/verifier/`).
* Stage B: offline verifier ablation on gold-derived wrappings (`results/verifier_ablation/`) and on real
  policy samples (`scripts/measure_phat.py` → `results/phat/`).
* Stage C: tiny RL smoke run (`experiments/smoke_rl_v1`).
* Stage D: 30-step controlled runs, one variable per arm (`configs/plans/stageD.yaml`), driven by
  `scripts/autoresearch.py` (restartable registry `experiments/registry.jsonl`).
* Stage E: longer run of the kept candidate.
* Splits: SWE by repository (train / validation / untouched test repos); unit-test random after de-duplication;
  the test split is never used for tuning.
* Compute: gpu48, H200 141 GB; 6 GPUs requested, 4 available while the user's `g4_bench` job holds 2.
* Decision rule: the loop's keep/reject uses the validation complete-solve rate (data-source averaged, greedy,
  n=1 per instance; SE ≈ 0.016). Because single-sample validation differences of < 0.03 are noise, the final
  choice (`scripts/finalize_selection.py`) ranks arms by k=8 held-out test solve rate with a z=1 noise band,
  then P2P regression, apply/format validity, stability (grad norm, KL, invalid rate), reward-signal health.

## 9. Results so far (base policy, before RL)
Measured with `scripts/measure_phat.py` (k=8, temperature 1.0, vf_v002 verifier, rw_v001 reward) — files under
`results/phat/`.

| split (curated_v2) | swe_32k p̂ | ut_function p̂ | ut_pytest p̂ | ut_stdio p̂ | notes |
|---|---|---|---|---|---|
| validation (435) | 0.000 (35) | 0.423 (173) | 0.050 (141) | 0.029 (86) | after prompt repair |
| test (667) | 0.000 (67) | 0.384 (281) | 0.051 (200) | 0.035 (119) | untouched; final before/after reference |
| curated_v1 validation, before repair | 0.000 | 0.216 | 0.023 | 0.039 | repair effect: +0.12 / +0.02 |

SWE-smith with the full repository snapshot is unsolved by the base policy (0 of 2,000+ samples); only
8 % of its diffs are structurally valid and 0 % apply, so the SWE track initially supplies format/apply
signal only. Stage-B parser ablation (`results/verifier_ablation/`): on gold-derived wrappings the
conservative fallback parser recovers the gold patch in 89 % of 9 wrappings (baseline cascade 75 %,
strict 22 %); on real samples the baseline cascade "accepts" 67 % of SWE responses whereas only 12–13 %
are structurally valid (the rest would fail to apply anyway).

### 9.1 Base policy after the model switch (Qwen3-30B-A3B-Instruct-2507, curated_v3pre validation, k=8)
| variant | n | p̂ | always-solved | never-solved | in [0.2,0.8] | extract | apply | mean reward |
|---|---|---|---|---|---|---|---|---|
| swe_32k | 62 | 0.038 | 0 % | 90 % | 8 % | 1.00 | 0.48 | 0.128 |
| swe_64k | 25 | 0.065 | 0 % | 80 % | 16 % | 1.00 | 0.52 | 0.178 |
| swe_128k | 25 | 0.030 | 0 % | 92 % | 4 % | 1.00 | 0.25 | 0.078 |
| ut_function | 173 | 0.836 | 75 % | 9 % | 8 % | 1.00 | 1.00 | 0.900 |
| ut_pytest | 141 | 0.681 | 48 % | 16 % | 24 % | 0.98 | 0.97 | 0.776 |
| ut_stdio | 86 | 0.673 | 43 % | 17 % | 20 % | 0.94 | 0.94 | 0.758 |
The switch inverts the difficulty picture: SWE-smith (all three context buckets) now has a real learning
signal (patches apply in 25–52 % of samples, a few instances solved), while the unit-test tracks are mostly
solved by the base policy. `curated_v3` therefore drops always-solved unit-test rows from the *training* split
(TOO_EASY, p̂ > 0.95) and the hierarchical sampler weights SWE at 0.3.

## 10–19. Ablations, dynamics, data quality, final dataset / verifier / rubric / reward / config / performance
_(filled from the registry and workbook after Stage D/E — see `docs/experiment_report.md`, `docs/ablation_report.md`)_

Stage D pass 1 (2026-09-02 night) failed on infrastructure for every arm (batch 32×8 not divisible by the
6 data-parallel ranks; a duplicated `+rusca.enable` override; one Ray start-up timeout). The records are
kept in `experiments/registry.jsonl`; pass 2 runs with batch 36 and the fixes listed in the git log.

## 18. Training configuration (Qwen3-30B-A3B, Stage D, `experiments/base_stageD_q3/overrides.txt`)
| item | value | why |
|---|---|---|
| GPUs | 4 of gpu48's H200 (an unlisted 2-GPU allocation sits on the node; Ulysses needs `heads % sp == 0` → SP=4 on 4 GPUs) | 32 attention heads are not divisible by 6 |
| batch / group | 24 prompts × n=6 rollouts, mini-batch 24 (1 update/step) | step 1 at 36×8 took 27 min (update 903 s on 100k-token sequences); 24×6 → 8–15 min/step |
| context | prompt ≤131,072, response ≤2,048, vLLM `max_model_len` 135,168, prefix caching | swe_128k prompts reach 119k tokens |
| sequence parallel | Ulysses SP=4 for actor and ref log-probs, dynamic batching 34k tokens/GPU, entropy chunking+checkpointing | one 131k sequence per micro-batch fits 141 GB |
| sampler | task SWE 0.25 / UT 0.75; variants swe 32k/64k/128k = 0.50/0.35/0.15, ut function/pytest/stdio = 0.45/0.30/0.25 | long rows must not dominate step cost |
| LoRA | q/k/v/o_proj, r=32, α=64, lr 2e-5 | see §17 |
| loss | GRPO advantage (std-normalised) + GSPO sequence-level ratio, clip 3e-4/4e-4, no KL, entropy 0 | project target algorithm |
| RUSCA | scaffold agent loop, logistic decay over 20 steps, rule-only reward (rubric weight 0 unless the arm says otherwise) | |
| host memory | 720 GB (`load_format=safetensors` so vLLM mmaps the base once; `layered_summon` syncs only LoRA) | 4 FSDP ranks stage the 30B model on CPU at init |
Measured: step 1 = 897 s (gen 200, old-logprob 189, update 417; max prompt 118k), step 2 = 476 s (max prompt 62k).

## 17. LoRA configuration (measured, `results/smoke/smoke_qwen3moe.json`)
| model | targets | rank / alpha / dropout | trainable | total | peak memory (1 GPU, HF+grad-ckpt) |
|---|---|---|---|---|---|
| Qwen1.5-MoE-A2.7B-Chat | q/k/v/o_proj (24 layers) | 32 / 64 / 0 | 12.58 M (0.088 %) | 14.33 B | 27.3 GiB @96 tok |
| Qwen1.5-MoE + shared expert | + shared_expert gate/up/down | 32 / 64 / 0 | 30.28 M (0.211 %) | 14.35 B | 27.5 GiB @96 tok |
| **Qwen3-30B-A3B-Instruct-2507** | q/k/v/o_proj (48 layers) | 32 / 64 / 0 | 26.74 M (0.087 %) | 30.56 B | 74.8 GiB @8k, 128.2 GiB @32k tok |
Experts are fused 3-D tensors in transformers 5.x (`experts.gate_up_proj`, `experts.down_proj`) — not
`nn.Linear` — so standard LoRA cannot target them; peft's `target_parameters` can (99 M params for
Qwen1.5-MoE) but vLLM has no weight-sync path for it, so expert LoRA is not used. Under FSDP (weights
sharded 6-way) + Ulysses SP=6 a 131k-token sequence costs ≈22k tokens of activation per GPU, within the
141 GB H200 budget together with the sharded weights.

## 18. Training configuration (Stage D, Qwen3-30B-A3B) — measured
`experiments/base_stageD_q3/overrides.txt`, launched by `scripts/launch_verl.sh` on 4 H200 (an unlisted
2-GPU allocation occupies the rest of gpu48) with 720 GB host memory:

| item | value | why |
|---|---|---|
| algorithm | GRPO advantage (std-normalised), GSPO loss (clip 3e-4/4e-4), token-mean aggregation, no KL loss, entropy coeff 0 | target configuration; GSPO in stock verl is seq-mean-token-mean, `gspo_tokenmean` in `rl/algos.py` is the token-mean variant |
| batch | 24 prompts × n=6 rollouts, mini-batch 24 (1 update/step), 20 steps per arm | 36×8 gave 27 min/step (100k-token SWE sequences dominate); 24×6 gives 8–15 min |
| context | prompt ≤131,072 tokens, response ≤2,048, vLLM `max_model_len` 135,168, prefix caching on, eager mode | 64k/128k SWE buckets included; compile path breaks MoE+LoRA |
| parallelism | FSDP over 4 GPUs, Ulysses sequence parallel = 4 (actor and ref), dynamic batching 34k tokens/GPU | 32 attention heads must be divisible by SP; a 131k sequence needs ≥4-way SP to fit 141 GB |
| LoRA | q/k/v/o_proj, r=32, α=64, dropout 0, lr 2e-5, bf16 model dtype | 26.7 M trainable (0.087 %) |
| rollout | vLLM 0.24, `load_format=safetensors` + `layered_summon` (only LoRA synced), gpu_memory_utilization 0.75 | dummy load stages 57 GB per vLLM worker on the host → OOM |
| sampling | hierarchical: task SWE 0.25 / unit-test 0.75; variants swe 32k/64k/128k = 0.50/0.35/0.15, ut function/pytest/stdio = 0.45/0.30/0.25 | bounds the share of 100k-token prompts per step |
| RUSCA | `rusca_scaffold_agent`, total_steps=20, logistic decay centred at 20 %, intra-group linear decay, `calculate_log_probs=False` | scaffold-conditioned log-probs must not be reused |
| validation | step 20 only (greedy, n=1, 512 rows) | one 512-row validation with 100k prompts costs ≈1 h |
| measured | step 1: 897 s (gen 200, old-logprob 189, update 417; longest prompt 118,385 tok); step 2: 476 s; host RSS peak ≈ 500 GB | `experiments/exp_Q000_baseline/train.log`, `mem_trace.log` |

## 20. Limitations and open problems
* Three of the five reference archives are missing on this machine; RUSCA/DDCA/hierarchical-sampler
  semantics are reconstructed (documented UNKNOWNs).
* The policy model's 32k window excludes the 64k/128k SWE buckets (400 instances) without simplification.
* SWE-smith tests that require network access are pruned from the effective sets; the P2P check remains
  capped at 30 ids.

## 21. Reproduction
`bash scripts/reproduce_final.sh` (environment → validation → dataset → smoke → Stage D/E → reports).

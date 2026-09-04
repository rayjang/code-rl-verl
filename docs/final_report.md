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

## 20. Limitations and open problems
* Three of the five reference archives are missing on this machine; RUSCA/DDCA/hierarchical-sampler
  semantics are reconstructed (documented UNKNOWNs).
* The policy model's 32k window excludes the 64k/128k SWE buckets (400 instances) without simplification.
* SWE-smith tests that require network access are pruned from the effective sets; the P2P check remains
  capped at 30 ids.

## 21. Reproduction
`bash scripts/reproduce_final.sh` (environment → validation → dataset → smoke → Stage D/E → reports).

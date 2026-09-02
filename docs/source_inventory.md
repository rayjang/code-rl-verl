# Source inventory

Generated 2026-09-02 on glogin02 / gpu48 (KISTI Neuron). All raw archives stay immutable at the
project root; extracted copies live under `sources/` (git-ignored).

## 1. Requested archives — what exists on this machine

| Archive | Requested role | Found? | SHA256 | Extracted to |
|---|---|---|---|---|
| `t01_math_kit.tar.gz` (`T15_longctx/T01_MATH_SCORE_LOCATIONS.md`) | math reward locations / scorer contract | **NOT FOUND** | – | – |
| `t15_rl_bundle.tar.gz` (`modules/T15_longctx/ALGO_CODE_MAP.md`) | RL algorithm map (GRPO/GSPO/DDCA/RUSCA/hier sampler) | **NOT FOUND** | – | – |
| `t01_math_reward_rubric.zip` | math rubric / reward reference | yes | `eb8c0a45d01cdf37223f8fd4afe053de4cdb3cf6d317698d8786e7e4bb7ca18b` | `sources/t01_math_reward_rubric/` |
| `rl_code_v1.zip` | existing coding RL dataset + reward + env kit | yes | `51aa813805b6b1a5e4af43afa370f37d06e5a1a42e8158faf10d467e821be7b2` | `sources/rl_code_v1/` |
| `swe_code_kit.tar.gz` (≈802 swe-smith environments) | execution environment reference | **NOT FOUND** | – | – |

Search performed: `find /scratch/r919a03 /home01/r919a03 -maxdepth 6` for `*math_kit*`, `*rl_bundle*`,
`*code_kit*`, `ALGO_CODE_MAP.md`, `T01_MATH_SCORE_LOCATIONS.md`, `*rusca*`, `*swe_smith*`, `T15_longctx`,
plus shell history. Only the two archives above exist. `rl_code_v1/reward/t15_reward_dispatch.py:34`
hard-codes the original bundle location as `T15_DIR=/scratch/r919a13/data/T15_longctx` (another user's
scratch); that directory was **not accessed** (permission/ownership boundary). If the owner can provide
`t15_rl_bundle.tar.gz` / `t01_math_kit.tar.gz` / `swe_code_kit.tar.gz`, the UNKNOWN items in
`docs/rusca_reverse_engineering.md` can be closed.

Consequence: RUSCA / DDCA / hierarchical-sampler / multi-task-reward semantics are reconstructed from
secondary evidence (the two present archives, which quote the missing kit) and marked UNKNOWN where
unverifiable.

## 2. Contents of the present archives

### 2.1 `rl_code_v1.zip` (44 files, 188 MB)

| Path | Role |
|---|---|
| `HANDOFF.md`, `README.md`, `build_report.json` | hand-off notes: 23,142 training instances (SWE-smith 751 + unit-test 5,927 core + 16,464 extended), image lists, reward ladder, known-unverified items |
| `docs/REWARD_DESIGN.md` | the 4 reward decisions (extraction cascade, partial credit `0.10+0.60·frac^1.5`, format vs logic failure bands, P2P multiplier `1−0.75·reg^1.5`), anti-hacking findings (`__eq__` canary) |
| `docs/ENVIRONMENT.md` | image preparation, isolation flags, patch-direction trap, pytest/ANSI traps, cost figures |
| `docs/MATH_REWARD.md` | math scorer contract (`score`, `answer_match`, `score_before_overlong`), **RUSCA section 4** (scaffold decay table, rank order, rule-only transition at ~24 %) |
| `reward/patch_extract.py` | 6-stage extraction cascade (fenced diff → any fence → raw git → raw unified → `*** Begin Patch` → SEARCH/REPLACE synthesis) |
| `reward/t15_code_judge.py` | `score_of()` ladder, forbidden test-path check, index loading |
| `reward/swe_exec_v2.py` | singularity 2-pass runner (git archive branch → restore tests from main → apply → `pytest -rA`) |
| `reward/unittest_exec.py` | function / pytest / stdio harnesses, rlimits, `__eq__` canary |
| `reward/t15_code_reward.py` | verl contract (24 keys), overlong penalty, `infra_excluded` |
| `reward/t15_reward_dispatch.py` | multi-task dispatcher with key-union template (wraps the missing `t15_reward_multi.py`) |
| `reward/async_wrap.py` | executor + semaphore + per-sample timeout wrapper |
| `harness/*.py` | contract/ladder harness, `measure_phat.py` (pass@k), synthetic swe-smith fixture |
| `env/*` | `unittest.def`, `Dockerfile.unittest`, image pull/bootstrap/retarget scripts, `verify_images.py` |
| `data/index/*.jsonl` | 751 SWE-smith, 5,927 + 16,464 unit-test, 120 R2E (excluded) instances (full + learnable views) |
| `data/parquet/*.parquet` | verl inputs: core train 6,516 / core val 162 / extended train 16,464 |
| `data/index/images_manifest.json` | 19 `jyangballin/swesmith.x86_64.*` images with digests (24.0 GiB compressed) |

### 2.2 `t01_math_reward_rubric.zip`

| Path | Role |
|---|---|
| `README.md` | RuscaRL hand-off for math: rubric schema, `index = uid` join, `rusca_scaffold_agent` launch flags, measured rubric effect |
| `t01_math_reward.py` (54 KB) | rule-based math scorer, 21-key contract |
| `rubrics/rubric_math.jsonl` (4,911 rows) | rubric records `{index, criteria[{criterion, points, tags{judge,inject,scaffold_rank,scaffold_kind,verify,reward}}]}` |

## 3. Runtime inventory (verified 2026-09-02)

| Item | Value |
|---|---|
| Login node | glogin02; compute node `gpu48` via SLURM partition `koni`, mandatory `--comment="field=nlp;appl=pytorch"` |
| GPUs | 8× H200 141 GB on gpu48; 2 held by another user; jobs request `--gres=gpu:6` → `CUDA_VISIBLE_DEVICES=0..5` (physical 2–7) |
| Container runtime | singularity-ce 4.3.4 (no fakeroot/subuid → no `.def` builds; images pulled from Docker Hub) |
| Python env | `.venv` (uv, CPython 3.12.12): torch 2.11.0+cu130, vLLM 0.24.0, verl 0.9.0, transformers 5.10.4, peft 0.20.0, flash-attn 2.8.3 (built from source, sm90), TransferQueue 0.1.8 |
| Model | `Qwen/Qwen1.5-MoE-A2.7B-Chat` snapshot `ec052fda…` (14.3 B params, 60 experts / 4 active + shared expert, 32 k context) |
| Images | `environments/sif/`: 19 swesmith `.sif` (23 GB total) + `python_3.11-slim-bookworm.sif` (unit-test sandbox) with host venv `environments/ut_venv` |

## 4. Dependencies between the materials

```
rl_code_v1 ──quotes──▶ T01_MATH_SCORER_LOCATIONS.md (missing kit)   contract keys, RUSCA profile
rl_code_v1 ──wraps───▶ t15_reward_multi.py (missing bundle)          multi-task dispatcher
rl_code_v1 ──follows─▶ swe_code_kit conventions (missing)            image/branch rules, SWE_EXEC_PATHS.md
t01_math_reward_rubric ──same schema──▶ rl_code_v1 parquet extra_info.rubrics (code profile)
t01_math_reward_rubric ──launch flags─▶ rusca_scaffold_agent / rusca_agent_loop.yaml (missing bundle)
```

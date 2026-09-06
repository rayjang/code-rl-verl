# code-rl-verl — coding-RL reward / verifier / rubric research on verl

GRPO/GSPO training of a coding policy on **verl 0.9.0**, where the reward comes from *actually running
the tests* inside a per-instance Singularity container — not from a string match.

Two task tracks share one trainer:

| track | `data_source` | what the policy produces | how it is scored |
|---|---|---|---|
| repo patch | `t15_repo_patch` | a diff against a repo snapshot (18k–119k-token prompts) | apply the patch in a SWE-smith image, run F2P + P2P pytest suites |
| unit-test impl | `t15_unittest_impl` | a function / module implementation | run hidden tests in a `python:3.11-slim` sandbox |

The reward is a configurable composition of an F2P term, a P2P regression term, rule checks, an
anti-reward-hacking inspection and an optional rubric term; 16 reward configs (`configs/reward/rw_v0*.yaml`)
are the arms of the ablation whose results are committed under `experiments/`.

---

## 1. What is in this repository

```
rl/                 verl entrypoint + hierarchical sampler + RUSCA agent loop + custom algos
verifier/           the reward function: container runners, diff parser, F2P/P2P, rubric, anti-hacking
configs/            reward/ verifier/ rubric/ agent-loop configs, and autoresearch plans
rubrics/            rubric definitions
scripts/            dataset build, image pull, SLURM launchers, evaluation, reporting
data/curated/       training parquets (train/validation/test) + manifests + audits
data/processed/index/  per-instance index the verifier reads at scoring time  ($RL_INDEX_DIR)
environments/       images list, validation manifests, ut_venv (bind-mounted pytest sandbox)
experiments/        one directory per arm: env.sh, overrides.txt, train.log, metrics.json
docs/               design + reproduction reports
tests/              unit tests for the verifier
```

Not in git, and why:

| path | size | how to get it |
|---|---|---|
| `environments/sif/*.sif` | 23 GB | `scripts/pull_images.sh` or `scripts/fetch_sif_release.sh` — §2.3 |
| `.venv/` | — | build it — §2.2 |
| `checkpoints/`, `release/*.zip` | 400 GB+ | training output, regenerate |
| `environments/cache*/` | 18 GB | scratch, regenerated on each run |
| `sources/` | — | the immutable input archives; everything derived from them **is** committed |

---

## 2. Setup

### 2.1 Requirements

Linux x86_64, ≥4 GPUs (the committed runs used 6× on one node), Singularity/Apptainer, Python 3.12.
The exact stack the results were produced with:

```
torch 2.11.0+cu130   vllm 0.24.0   verl 0.9.0
transformers 5.10.4  peft 0.20.0   flash_attn 2.8.3
ray 2.58.0           datasets 5.0.1  tensordict 0.10.0
```

`requirements-lock.txt` is a full 250-package snapshot of that environment. It is a reference, not a
one-shot installer — torch, vllm and flash-attn need their own index or a local build.

### 2.2 Python environment

```bash
git clone https://github.com/rayjang/code-rl-verl.git && cd code-rl-verl
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install vllm==0.24.0 verl==0.9.0 transformers==5.10.4 peft==0.20.0 \
               ray==2.58.0 datasets==5.0.1 tensordict==0.10.0 torchdata==0.11.0 \
               hydra-core==1.3.6 omegaconf==2.3.1 pandas pyarrow pyyaml codetiming
bash scripts/build_flash_attn.sh          # flash-attn 2.8.3, compiled against the installed torch
python -m pytest tests/verifier -q        # verifier unit tests: 55 passed, no GPU and no images needed
```

### 2.3 Container images (23 GB, required for any reward that runs tests)

Either pull them from Docker Hub (public, no token, reproduces the images bit-for-bit):

```bash
sbatch scripts/pull_images.sh             # or: bash scripts/pull_images.sh
```

or download the prebuilt `.sif` from this repo's release:

```bash
GITHUB_TOKEN=<pat> bash scripts/fetch_sif_release.sh
```

Both write to `environments/sif/` and skip images that are already there.

### 2.4 Unit-test sandbox

```bash
sbatch scripts/build_ut_sandbox.sh        # populates environments/ut_venv (bind-mounted into the container)
```

---

## 3. Train on the committed dataset

An experiment is a directory with two files. `env.sh` picks the model and the reward/verifier/rubric
configs; `overrides.txt` is a list of Hydra overrides for verl.

```bash
# experiments/my_run/env.sh
export EXP_NAME=my_run
export MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507   # any HF snapshot dir
export DATASET_VERSION=curated_v3                        # selects data/processed/index/<version>
export REWARD_CONFIG=$ROOT/configs/reward/rw_v001_baseline.yaml
export VERIFIER_CONFIG=$ROOT/configs/verifier/vf_v002.yaml
export RUBRIC_CONFIG=$ROOT/rubrics/rubric_v001_code_hint.yaml
export VERIFIER_CONCURRENCY=6
```

```bash
cp experiments/exp_Q000_baseline/overrides.txt experiments/my_run/overrides.txt
sbatch scripts/launch_verl.sh experiments/my_run     # SLURM
# or directly:
ROOT=$PWD NGPU=6 bash scripts/launch_verl.sh experiments/my_run
```

`launch_verl.sh` resolves `$ROOT/$MODEL_PATH/$EXP_NAME/$NGPU` inside the overrides, writes
`overrides.resolved.txt`, records the environment in `run_info.txt`, traces host RSS into
`mem_trace.log`, and runs `python -m rl.main_ppo_code`. Per-sample verifier output goes to
`instance_log.jsonl`.

> The SLURM header in `launch_verl.sh` is site-specific (`-p koni -w gpu48 --comment=...`, KISTI Neuron).
> Edit the `#SBATCH` lines for your cluster, or run the script directly with `bash`.

Smoke test first — 20 steps on a tiny split, no SWE images needed:

```bash
python scripts/prepare_smoke_data.py
sbatch scripts/launch_verl.sh experiments/smoke_rl_v1
```

---

## 4. Train on **your own** coding data

The trainer reads two parquets; the verifier reads a JSONL index keyed by `instance_id`. You need both,
because the parquet holds the *prompt* and the index holds the *tests*.

### 4.1 Parquet (what verl loads)

`data.train_files` / `data.val_files` in `overrides.txt`. One row per problem:

| column | value |
|---|---|
| `data_source` | `t15_unittest_impl` or `t15_repo_patch` — selects the runner |
| `prompt` | chat turns: `[{"role": "system", ...}, {"role": "user", ...}]` |
| `ability` | free-form tag, e.g. `swe_patch_longctx` |
| `reward_model` | `{"style": "rule", "ground_truth": "<instance_id>"}` — **the join key** |
| `extra_info` | `{"index": "<instance_id>", "split": ..., "task_tag": "code", "rubric_profile": "code", "rubrics": [...], "variant_type": ...}` |

`variant_type` is what the hierarchical sampler balances on (`+data.hier_sampler.variant_weights.*` in
the overrides); use your own names and weight them there.

### 4.2 Index (what the verifier loads)

Files under `$RL_INDEX_DIR` (= `data/processed/index/$DATASET_VERSION`), read by `verifier/registry.py`:

- `t15_code_unittest*.full.jsonl` — one line per unit-test instance: `instance_id`, `harness`
  (`function` | `pytest` | `stdio`), `tests` (list of `{assertion, ...}`), `entry_point`
- `t15_code_swesmith*.full.jsonl` — one line per repo instance: `instance_id`, `f2p`/`p2p` test node ids,
  `test_patch`, repo metadata
- `t15_code_*.learnable.jsonl` — the subset actually trained on; SWE lines carry `image_name`, which the
  registry rewrites to `$RL_SIF_DIR/<image>.sif`

Look at the committed `data/processed/index/curated_v3/*.jsonl` for the exact field set — it is the
authoritative spec, and one line of it is a complete example.

### 4.3 Bringing your own repositories

For the unit-test track nothing else is needed: the sandbox is the stock `python:3.11-slim` image plus
`environments/ut_venv`, so new problems only mean new index lines.

For the repo-patch track each instance needs an image containing the repo at the broken commit. Add the
image tags to `environments/images/images_t15_code_swesmith.txt`, run `scripts/pull_images.sh`, and set
`image_name` on the index lines accordingly.

### 4.4 Check it before you train

```bash
export RL_INDEX_DIR=$PWD/data/processed/index/my_version RL_SIF_DIR=$PWD/environments/sif
python scripts/validate_environments.py          # every instance's tests actually run in its image
```

This writes `environments/manifests/validation_{swe,ut}.jsonl` — the build step drops instances whose
environment is broken, so a bad image fails here rather than silently scoring 0 for 20 steps.

### 4.5 Rebuilding the curated split

`scripts/build_dataset.py` is the pipeline that produced `curated_v1…v3`: it joins raw parquets with the
index and the validation manifests, drops instances with reason codes, optionally filters by measured
base-policy difficulty (`--phat`, from `scripts/measure_phat.py`), and writes both the parquets and the
augmented index.

```bash
python scripts/build_dataset.py --version my_version --seed 0 \
       --phat results/phat/<run>/phat.jsonl --band 0.05,0.95
```

---

## 5. Reward configuration

`REWARD_CONFIG` selects one of `configs/reward/rw_v0*.yaml`:

| config | arm |
|---|---|
| `rw_v001_baseline` | the default composition |
| `rw_v002…v005` | F2P shaping: binary / linear / sqrt / ladder |
| `rw_v006…v010` | P2P regression penalty: strong / tolerant / capped / confidence / none |
| `rw_v011_rule_only` | no rubric, rules only |
| `rw_v012_rubric_heavy` | rubric-weighted |
| `rw_v013_rusca` | RUSCA scaffold schedule |
| `rw_v014/v015` | additive, gate off / on |
| `rw_v016_no_apply_credit` | no partial credit for a patch that applies |

The reward function itself is `verifier/verl_reward.py:compute_score`, wired in through
`reward.custom_reward_function.path`. It reads these env vars, all optional:

```
RL_INDEX_DIR  RL_SIF_DIR  VERIFIER_CONFIG  REWARD_CONFIG  RUBRIC_CONFIG
RUSCA_STEP_FILE  VERIFIER_RUN_DIR  VERIFIER_CONCURRENCY  VERIFIER_LOG_JSONL
```

Infrastructure failures never leak into the gradient: they return `infra_excluded=1` and the configured
`infra_score`.

---

## 6. Results and reports

`experiments/exp_D*` (Qwen1.5-MoE-A2.7B-Chat) and `experiments/exp_Q*` (Qwen3-30B-A3B-Instruct-2507) hold
the committed arm runs — `train.log`, `metrics.json`, resolved overrides, per-sample `instance_log.jsonl`.

```bash
python scripts/report_experiments.py                          # arm comparison
python scripts/report_dataset_quality.py --version curated_v3
python scripts/build_excel.py                                 # results/rl_code_reward_research.xlsx
```

Write-ups: `docs/final_report.md`, `docs/experiment_report.md`, `docs/ablation_report.md`,
`docs/coding_reward_design.md`, `docs/verifier_design.md`, `docs/rubric_design.md`,
`docs/dataset_quality_report.md`.

End-to-end reproduction from the raw archives: `scripts/reproduce_final.sh` (8 idempotent stages).

---

## 7. Datasets

| version | rows (train) | note |
|---|---|---|
| `curated_v1` | 5,146 | first curated build |
| `curated_v2` | 4,507 | + measured base-policy difficulty band |
| `curated_v3pre` | 5,423 | Qwen3 policy; all 504 environment-valid SWE rows kept |
| `curated_v3` | 2,074 | v3pre + Qwen3 base-policy difficulty filter on the train split |
| `curated_v4_single` | 6,634 | single-file, difficulty-annotated (Qwen3 k=8) — see `data/curated/curated_v4_single/DATASET.md` |

Every build ships `dataset_manifest.json` (counts, seeds, exclusion reason codes) and
`instance_audit.csv` next to the parquets.

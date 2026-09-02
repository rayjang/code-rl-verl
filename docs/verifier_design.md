# Verifier design (coding RL)

Package `verifier/` (unit tests in `tests/verifier/`, 52 tests). Every stage returns a typed record
(`verifier/schemas.py`) and every response gets one of the required statuses:
`no_patch → diff_parse_success | diff_parse_fail → git_apply_check_success → patch_apply_success | patch_apply_fail`.

## 1. Patch extraction (`verifier/diff_parser.py`)

| Candidate | Rule | Repair risk |
|---|---|---|
| `strict` | whole response (minus one optional ```` ```diff ```` fence) must be a valid unified diff; prose before/after → `diff_parse_fail` | none |
| `markdown` | last ```` ```diff/```patch ```` fenced block, structurally validated; unfenced diff → `diff_parse_fail` | none |
| `fallback` (conservative) | markdown → any fence containing a diff signature → raw `diff --git` / `--- a/` region trimmed at the last complete hunk; validated | none (trailing prose trimmed only at a hunk boundary) |
| `baseline` | port of `rl_code_v1/reward/patch_extract.py` (6-stage cascade, no validation, `*** Begin Patch` conversion fabricates hunk headers and relies on `patch --fuzz=5`) | **yes** – kept only as an ablation arm and flagged `repair_risk` |

Allowed normalisation: CRLF→LF, fence stripping, trailing newline, trailing prose after the last hunk.
Forbidden: fixing hunk counts, inventing headers/line numbers, fuzzy path matching.
SEARCH/REPLACE blocks (the format the dataset's *system prompt* asks for) are converted to a diff only
when the instance's `buggy_files` are known and every SEARCH matches exactly once (`fmt=search_replace`);
that is a format conversion, not a repair. Selection among several candidates: `last` (baseline
rationale), `first`, `longest`, `concat_distinct` are all supported for the ablation.
Unit-test track: last ```` ```python ```` block; `compile()` failure = `diff_parse_fail`.

## 2. Patch application (`verifier/patch_apply.py`)
Host-side `git apply --check` then `git apply` on the extracted tree (no git repository → strict patch
tool, no fuzz). `mode=fuzz` (baseline `patch --fuzz=5` fallback) exists for the ablation only and is
reported as a repair. `ignore_whitespace=True` (`git apply --ignore-whitespace`) is the default of
`vf_v002` because the prompt snapshots (`buggy_files`) differ from the repository tree in whitespace for
31/751 instances (their **gold** patches fail strict apply); the model only ever sees the snapshot.

## 3. Execution
* **SWE-smith runner** (`verifier/swe_runner.py`): per instance a cached base tree = copy of the image's
  `/testbed` (keeps untracked build artifacts such as generated `_version.py`), `git checkout -f origin/<iid>`
  (buggy state), test material restored from `main` (test dirs, test-pattern files, root `conftest.py`;
  never source modules that appear as doctest ids), `.git` removed, tarred. Evaluation: untar → host-side
  apply → `singularity exec --containall --no-home --cleanenv --net --network=none --writable-tmpfs
  --bind tree:/testbed` → `python -m pytest -rA --tb=no --color=no -p no:cacheprovider [--doctest-modules] <ids>`
  with a hard `timeout -s KILL`. Parsed with ANSI stripping; parametrised ids pass only if all parameters pass.
* **Unit-test runner** (`verifier/ut_runner.py` + in-container `verifier/ut_driver.py`): one isolated
  container launch per rollout (python:3.11-slim + read-only host venv with pytest/numpy/sympy),
  per-assertion subprocesses with rlimits (AS/DATA/NPROC/FSIZE), function / pytest / stdio harnesses,
  network disabled (verified), always-true-`__eq__` canary (AST-rewritten assertions + sentinel probes).

### 3.1 pytest-crash rule
If pytest cannot start on the *patched* tree (no `collected N items` / session banner) the failure is attributed
to the patch (`hack_flags=pytest_crashed`, every F2P/P2P id failed) because every curated instance's environment
was validated with the gold patch. Only image/pytest-missing signatures remain `infra_env`. Rationale:
13 `agronholm__exceptiongroup` instances inject the bug into a package that pytest itself imports, so an
unfixed tree cannot run tests at all — that is exactly the behaviour a policy must learn to repair.

## 4. Effective test sets (from `scripts/validate_environments.py`)
For every SWE instance the validation run records `f2p_effective` (ids that fail on the buggy tree AND pass
with gold) and `p2p_effective` (ids that pass both before and after gold). The runner scores only these ids
when present, so environment-failing tests (network, plugins, flaky) do not count as model failures or
regressions. Instances with an empty effective F2P are excluded (`NO_VALID_F2P`).
Validation v5 (2026-09-02): 666 `ok` + 52 `ok_effective` of 751 before the pytest-crash rule; 20 `gold_fail`
(11 inflect instances whose branch lacks the patched file, 7 python-string-similarity bugs injected into
`*_test.py`, 2 pygments), 13 exceptiongroup crash cases re-validated under the rule above.

## 5. Anti-hacking (`verifier/anti_hacking.py`)
Path classes (tests, config, verifier files, path escape) from the instance's own test ids plus patterns;
`allow_test_edits` metadata overrides the blanket rule; added-line patterns (`pytest.skip`, conftest hooks,
builtins monkeypatching, `os._exit`, symlinks, always-true `__eq__`, broad `except: pass`). Hard flags → `invalid_*`
(reward 0, not excluded), soft flags are logged (`hack_flags`).

## 6. Reward contract
`verifier/verl_reward.py::compute_score` returns a 33-key dict (same keys every call, float/str only):
all `RewardComponents` fields + `score`, `verify_runtime_s`, `rusca_step`. Infra failures → `infra_excluded=1`.
Per-sample instance-level records are appended to `$VERIFIER_LOG_JSONL`.

## 7. Versions
| verifier version | parser | select | apply | S/R | P2P cap | notes |
|---|---|---|---|---|---|---|
| vf_v001 | fallback | last | strict | on | 30 | baseline-equivalent extraction, strict apply |
| vf_v002 | fallback | last | strict + ignore-whitespace | on | 30 | default after environment validation |
(Additional parser/apply arms are produced by `scripts/verifier_ablation.py`.)

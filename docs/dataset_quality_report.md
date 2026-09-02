# Dataset quality report — `curated_v1`

Raw data (`sources/rl_code_v1`) is never modified; every exclusion below is recorded with a reason code in `data/curated/curated_v1/excluded.parquet` and `instance_audit.csv`.

## 1. Counts

| split | total | swe_32k | ut_function | ut_pytest | ut_stdio |
|---|---|---|---|---|---|
| train | 5146 | 227 | 2167 | 1677 | 1075 |
| validation | 435 | 35 | 193 | 115 | 92 |
| test | 667 | 67 | 260 | 220 | 120 |
| excluded | 430 | | |

SWE repositories per split (group split by repository): {"train": ["swesmith/Knio__dominate.9082227e", "swesmith/Suor__funcy.207a7810", "swesmith/agronholm__exceptiongroup.0b4f4937", "swesmith/agronholm__typeguard.b6a7e438", "swesmith/joke2k__faker.8b401a7d", "swesmith/luozhouyang__python-string-similarity.115acaac", "swesmith/oauthlib__oauthlib.1fd52536", "swesmith/pdfminer__pdfminer.six.1a8bd2f7", "swesmith/pudo__dataset.5c2dc8d3", "swesmith/pygments__pygments.27649ebb", "swesmith/pyupio__safety.7654596b"], "validation": ["swesmith/cantools__cantools.0c6a7871", "swesmith/PyCQA__flake8.cf1542ce", "swesmith/Cog-Creators__Red-DiscordBot.33e0eac7"], "test": ["swesmith/mahmoud__boltons.3bfcfdd0", "swesmith/cknd__stackprinter.219fcc52", "swesmith/jd__tenacity.0d40e76f"]}

## 2. Exclusions by reason code

| reason | count | meaning |
|---|---|---|
| CONTEXT_OVERFLOW | 400 | prompt does not fit the 32k context of Qwen1.5-MoE (swe_64k / swe_128k buckets) |
| NO_VALID_F2P | 15 | no F2P test both fails on the buggy tree and passes with gold |
| BUG_IN_TEST_FILE | 7 | injected bug lives in a test/doctest file that the smith convention restores from main (F2P passes with an empty patch or gold no longer applies) |
| DUPLICATE | 5 | identical prompt text already present |
| REFERENCE_SOLUTION_FAILS | 3 | unit-test reference solution fails its own tests |

## 3. Quality flags (kept instances, informational)

| flag | count | meaning |
|---|---|---|
| no_reference_solution | 1287 | stdio instances ship no reference (validated by empty-code check only) |
| prompt_gt_32k | 400 | prompt longer than 32k tokens |
| F2P_PRUNED | 79 | some F2P ids removed (fail with gold in this environment, e.g. network tests) |
| gold_touches_tests | 47 | gold edits a test-pattern file |
| P2P_PRUNED | 40 | some P2P ids removed (fail before any patch) |
| GOLD_WHITESPACE_MISMATCH | 29 | gold applies only with git apply --ignore-whitespace (prompt snapshot whitespace differs from the tree) |
| dup_of | 5 | duplicate of another instance |

## 4. Environment validation summary

* SWE-smith (751 instances, 19 images): {'gold_fail': 20, 'ok': 666, 'baseline_anomaly': 13, 'ok_effective': 52} — `ok` = empty patch fails all F2P and passes all P2P, gold resolves; `ok_effective` = usable after pruning environment-failing test ids.
* Unit-test track (22391 instances): {'ok': 22380, 'gold_fail': 6, 'baseline_anomaly': 5}.
* Details per instance: `environments/manifests/validation_swe.jsonl`, `validation_ut.jsonl` (earlier attempts kept as `*.v1_gitarchive`, `*.v2_copy`, `*.v3_fulltests`, `*.v4_testonly`).

## 5. Data defects found (and how they were handled)

1. **Prompt/format contradiction (all SWE rows)**: system prompt demands `<solution>` SEARCH/REPLACE blocks, the user turn demands a ```` ```diff ```` block. Handled in the verifier (both formats accepted); flagged for a prompt-fix ablation.
2. **Context overflow**: 64k/128k snapshot buckets cannot fit the policy's 32k window → excluded (CONTEXT_OVERFLOW), not simplified.
3. **Bug injected into test/doctest files** (47 instances, 2 repos): unsolvable under the smith convention → excluded.
4. **Whitespace drift between prompt snapshot and repository** (31 instances): handled by `--ignore-whitespace` apply (vf_v002), flagged.
5. **Environment-failing tests** (network access, plugin/config quirks): removed from the effective F2P/P2P sets instead of counting as model failures.
6. **Unit-test prompts that never name the required function/class** (audit: 84 % of pytest-harness statements): kept as-is for the baseline, measured through empirical p̂; see the difficulty section of `docs/final_report.md`.
7. **Unit-test reference solutions failing their own tests** (6) and no-op solutions passing tests (5): excluded.
8. **`judge_verdict=drop` rows** (430 SWE rows in the original core train): semantics undocumented; kept and flagged (JUDGE_DROP).

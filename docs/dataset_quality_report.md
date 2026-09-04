# Dataset quality report — `curated_v3`

Raw data (`sources/rl_code_v1`) is never modified; every exclusion below is recorded with a reason code in `data/curated/curated_v3/excluded.parquet` and `instance_audit.csv`.

## 1. Counts

| split | total | swe_128k | swe_32k | swe_64k | ut_function | ut_pytest | ut_stdio |
|---|---|---|---|---|---|---|---|
| train | 2074 | 86 | 177 | 178 | 403 | 707 | 523 |
| validation | 512 | 25 | 62 | 25 | 173 | 141 | 86 |
| test | 699 | 17 | 69 | 13 | 281 | 200 | 119 |
| excluded | 3393 | | |

SWE repositories per split (group split by repository): {"train": ["swesmith/Knio__dominate.9082227e", "swesmith/PyCQA__flake8.cf1542ce", "swesmith/Suor__funcy.207a7810", "swesmith/agronholm__typeguard.b6a7e438", "swesmith/bottlepy__bottle.a8dfef30", "swesmith/jaraco__inflect.c079a96a", "swesmith/jd__tenacity.0d40e76f", "swesmith/mahmoud__boltons.3bfcfdd0", "swesmith/oauthlib__oauthlib.1fd52536", "swesmith/pdfminer__pdfminer.six.1a8bd2f7", "swesmith/pudo__dataset.5c2dc8d3", "swesmith/pygments__pygments.27649ebb", "swesmith/pyupio__safety.7654596b"], "validation": ["swesmith/Cog-Creators__Red-DiscordBot.33e0eac7", "swesmith/cantools__cantools.0c6a7871", "swesmith/cknd__stackprinter.219fcc52"], "test": ["swesmith/joke2k__faker.8b401a7d", "swesmith/luozhouyang__python-string-similarity.115acaac", "swesmith/agronholm__exceptiongroup.0b4f4937"]}

## 2. Exclusions by reason code

| reason | count | meaning |
|---|---|---|
| TOO_EASY | 2816 | empirical success rate above the band |
| TOO_HARD | 533 | empirical success rate below the band |
| BUG_IN_TEST_FILE | 18 | injected bug lives in a test/doctest file that the smith convention restores from main (F2P passes with an empty patch or gold no longer applies) |
| NO_VALID_F2P | 16 | no F2P test both fails on the buggy tree and passes with gold |
| DUPLICATE | 5 | identical prompt text already present |
| REFERENCE_SOLUTION_FAILS | 3 | unit-test reference solution fails its own tests |
| BASELINE_ANOMALY | 2 | an empty/no-op solution already passes the tests |

## 3. Quality flags (kept instances, informational)

| flag | count | meaning |
|---|---|---|
| PROMPT_FIX_ENTRY_POINT | 4635 | prompt repaired: the names the hidden tests import/call were appended (82% of base-policy pytest failures were name mismatches) |
| no_reference_solution | 1287 | stdio instances ship no reference (validated by empty-code check only) |
| HARD_BUT_PARTIAL_SIGNAL | 516 | p_hat below the band but rollouts still show reward variance (partial F2P / format credit) -> kept |
| F2P_PRUNED | 92 | some F2P ids removed (fail with gold in this environment, e.g. network tests) |
| P2P_PRUNED | 53 | some P2P ids removed (fail before any patch) |
| gold_touches_tests | 47 | gold edits a test-pattern file |
| GOLD_WHITESPACE_MISMATCH | 29 | gold applies only with git apply --ignore-whitespace (prompt snapshot whitespace differs from the tree) |
| dup_of | 5 | duplicate of another instance |

## 4. Environment validation summary

* SWE-smith (751 instances, 19 images): {'gold_fail': 20, 'ok': 679, 'ok_effective': 52} — `ok` = empty patch fails all F2P and passes all P2P, gold resolves; `ok_effective` = usable after pruning environment-failing test ids.
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

## 6. Empirical difficulty (our policy, k samples per instance)

* swe_128k: n=93, mean p̂=0.048, p̂=0: 87.1%, p̂=1: 0.0%, in [0.2,0.8]: 11.8%
* swe_32k: n=198, mean p̂=0.117, p̂=0: 72.7%, p̂=1: 3.5%, in [0.2,0.8]: 13.6%
* swe_64k: n=213, mean p̂=0.066, p̂=0: 80.3%, p̂=1: 0.0%, in [0.2,0.8]: 12.7%
* ut_function: n=2166, mean p̂=0.824, p̂=0: 10.1%, p̂=1: 72.6%, in [0.2,0.8]: 10.6%
* ut_pytest: n=1671, mean p̂=0.669, p̂=0: 16.2%, p̂=1: 46.1%, in [0.2,0.8]: 24.2%
* ut_stdio: n=1082, mean p̂=0.661, p̂=0: 15.2%, p̂=1: 43.0%, in [0.2,0.8]: 26.7%

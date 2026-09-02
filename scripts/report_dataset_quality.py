#!/usr/bin/env python3
"""Render docs/dataset_quality_report.md from a curated dataset manifest + validation manifests + audit."""
import argparse, collections, json, os, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="curated_v1"); ap.add_argument("--out", default=f"{ROOT}/docs/dataset_quality_report.md")
    a = ap.parse_args()
    d = f"{ROOT}/data/curated/{a.version}"
    man = json.load(open(f"{d}/dataset_manifest.json"))
    audit = pd.read_csv(f"{d}/instance_audit.csv")
    vswe = jl(f"{ROOT}/environments/manifests/validation_swe.jsonl"); vut = jl(f"{ROOT}/environments/manifests/validation_ut.jsonl")
    L = []
    L.append(f"# Dataset quality report — `{a.version}`\n")
    L.append("Raw data (`sources/rl_code_v1`) is never modified; every exclusion below is recorded with a reason code in "
             f"`data/curated/{a.version}/excluded.parquet` and `instance_audit.csv`.\n")
    L.append("## 1. Counts\n")
    L.append("| split | total | " + " | ".join(sorted({k for v in man["counts"].values() if isinstance(v, dict) for k in v if k != "total"})) + " |")
    variants = sorted({k for v in man["counts"].values() if isinstance(v, dict) for k in v if k != "total"})
    L.append("|---|---|" + "---|" * len(variants))
    for s in ("train", "validation", "test"):
        c = man["counts"][s]; L.append(f"| {s} | {c['total']} | " + " | ".join(str(c.get(v, 0)) for v in variants) + " |")
    L.append(f"| excluded | {man['counts']['excluded']} | | |\n")
    L.append("SWE repositories per split (group split by repository): " + json.dumps(man["swe_repos"], ensure_ascii=False) + "\n")
    L.append("## 2. Exclusions by reason code\n")
    L.append("| reason | count | meaning |")
    L.append("|---|---|---|")
    meaning = {"CONTEXT_OVERFLOW": "prompt does not fit the 32k context of Qwen1.5-MoE (swe_64k / swe_128k buckets)",
               "BUG_IN_TEST_FILE": "injected bug lives in a test/doctest file that the smith convention restores from main (F2P passes with an empty patch or gold no longer applies)",
               "REFERENCE_PATCH_INVALID": "gold patch does not apply (even with --ignore-whitespace)",
               "BROKEN_BRANCH": "instance branch lacks the file the gold patch edits",
               "BROKEN_ENV": "base tree extraction / pytest could not run",
               "NO_VALID_F2P": "no F2P test both fails on the buggy tree and passes with gold",
               "BASELINE_ANOMALY": "an empty/no-op solution already passes the tests",
               "REFERENCE_SOLUTION_FAILS": "unit-test reference solution fails its own tests",
               "NO_TESTS": "instance has no tests", "DUPLICATE": "identical prompt text already present",
               "UNVALIDATED": "no validation record", "TOO_EASY": "empirical success rate above the band", "TOO_HARD": "empirical success rate below the band"}
    for k, v in sorted(man["exclusions"].items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v} | {meaning.get(k, '')} |")
    L.append("\n## 3. Quality flags (kept instances, informational)\n")
    L.append("| flag | count | meaning |"); L.append("|---|---|---|")
    fm = {"GOLD_WHITESPACE_MISMATCH": "gold applies only with git apply --ignore-whitespace (prompt snapshot whitespace differs from the tree)",
          "F2P_PRUNED": "some F2P ids removed (fail with gold in this environment, e.g. network tests)",
          "P2P_PRUNED": "some P2P ids removed (fail before any patch)", "JUDGE_DROP": "original judge_verdict=drop (semantics unknown, kept)",
          "gold_touches_tests": "gold edits a test-pattern file", "no_reference_solution": "stdio instances ship no reference (validated by empty-code check only)",
          "EMPTY_PASSES_SOME": "a no-op solution passes a minority of tests (partial-credit floor inflated)",
          "PROMPT_FIX_ENTRY_POINT": "prompt repaired: the names the hidden tests import/call were appended (82% of base-policy pytest failures were name mismatches)",
          "HARD_BUT_PARTIAL_SIGNAL": "p_hat below the band but rollouts still show reward variance (partial F2P / format credit) -> kept", "dup_of": "duplicate of another instance", "prompt_gt_32k": "prompt longer than 32k tokens"}
    for k, v in sorted(man["flags"].items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v} | {fm.get(k, '')} |")
    # env validation summary
    st = collections.Counter(r.get("status") for r in vswe); ut = collections.Counter(r.get("status") for r in vut)
    L.append("\n## 4. Environment validation summary\n")
    L.append(f"* SWE-smith (751 instances, 19 images): {dict(st)} — `ok` = empty patch fails all F2P and passes all P2P, gold resolves; "
             f"`ok_effective` = usable after pruning environment-failing test ids.")
    L.append(f"* Unit-test track ({len(vut)} instances): {dict(ut)}.")
    L.append("* Details per instance: `environments/manifests/validation_swe.jsonl`, `validation_ut.jsonl` (earlier attempts kept as `*.v1_gitarchive`, `*.v2_copy`, `*.v3_fulltests`, `*.v4_testonly`).\n")
    L.append("## 5. Data defects found (and how they were handled)\n")
    L.append("1. **Prompt/format contradiction (all SWE rows)**: system prompt demands `<solution>` SEARCH/REPLACE blocks, the user turn demands a ```` ```diff ```` block. Handled in the verifier (both formats accepted); flagged for a prompt-fix ablation.")
    L.append("2. **Context overflow**: 64k/128k snapshot buckets cannot fit the policy's 32k window → excluded (CONTEXT_OVERFLOW), not simplified.")
    L.append("3. **Bug injected into test/doctest files** (47 instances, 2 repos): unsolvable under the smith convention → excluded.")
    L.append("4. **Whitespace drift between prompt snapshot and repository** (31 instances): handled by `--ignore-whitespace` apply (vf_v002), flagged.")
    L.append("5. **Environment-failing tests** (network access, plugin/config quirks): removed from the effective F2P/P2P sets instead of counting as model failures.")
    L.append("6. **Unit-test prompts that never name the required function/class** (audit: 84 % of pytest-harness statements): kept as-is for the baseline, measured through empirical p̂; see the difficulty section of `docs/final_report.md`.")
    L.append("7. **Unit-test reference solutions failing their own tests** (6) and no-op solutions passing tests (5): excluded.")
    L.append("8. **`judge_verdict=drop` rows** (430 SWE rows in the original core train): semantics undocumented; kept and flagged (JUDGE_DROP).")
    if (audit["empirical_success_rate"].fillna(-1) >= 0).any():
        e = audit[audit["empirical_success_rate"] >= 0]
        L.append("\n## 6. Empirical difficulty (our policy, k samples per instance)\n")
        for vt, g in e.groupby("variant_type"):
            p = g["empirical_success_rate"]
            L.append(f"* {vt}: n={len(g)}, mean p̂={p.mean():.3f}, p̂=0: {(p == 0).mean():.1%}, p̂=1: {(p == 1).mean():.1%}, in [0.2,0.8]: {((p >= 0.2) & (p <= 0.8)).mean():.1%}")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

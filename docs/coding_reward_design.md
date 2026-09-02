# Coding reward design (RUSCA-based)

Implementation: `verifier/reward.py` (`RewardConfig`, `compute_reward`), transforms in `verifier/f2p.py`,
`verifier/p2p.py`; every candidate is a YAML in `configs/reward/` and is selected per experiment.

## 1. Principles
1. **Execution dominates.** The only way to reach reward 1.0 is `resolved` = every effective F2P test passes
   and no observed P2P regression. Rubric/scaffolding credit is bounded and, under RUSCA, decays to 0.
2. **Format failure ≠ logic failure.** Four disjoint states before execution (`no_patch`, `parse_fail`,
   `invalid`, `apply_fail`) with their own scores and their own `err_kind`; logic failure starts at the
   `applies` floor and grows with F2P credit.
3. **Infra failure is not the model's fault.** `infra_excluded=1`, score = `infra_score` (0), never counted
   as a solved/unsolved outcome in difficulty statistics.
4. **Every component is logged**: extraction, format, apply, F2P (raw + transformed), P2P (raw + multiplier),
   rubric, rule correctness, timeout penalty, length component, `score_before_overlong`, final.
5. **Config-driven.** Weights, transforms, gates and the RUSCA stage are fields of `RewardConfig`.

## 2. Families
### Ladder (baseline family, `mode: ladder`)
```
no_patch        -> s_no_patch (0.00)
parse_fail      -> s_parse_fail (0.00 baseline; 0.02 in the "format band" candidate)
invalid (hack)  -> s_invalid (0.00, not excluded)
apply_fail      -> s_apply_fail (0.05)
ran, all F2P, no regression         -> s_resolved (1.00)
ran, all F2P, regression            -> max(floor, s_f2p_all_regressed * M)          (0.85 * M)
ran, partial                        -> max(floor, (s_applies + w_partial * T(frac) + rw * rubric) * M)
T(frac): binary | linear | sqrt | pow(gamma) | ladder(thresholds)      M: P2P multiplier (below)
```
`rw_v001_baseline` = exact port of `t15_code_judge.score_of` (T=frac^1.5, M=1−0.75·reg^1.5); unit-tested
against the documented table (5/10 → 0.312, 29/30 → 0.846, 0/30 → 0.212).

### Additive (`mode: additive`)
`final = clip(w_fmt·fmt + w_apply·apply + rw·rubric + w_f2p·T + w_p2p·p2p_frac + bonus·resolved − pen·(1−M) − timeout, 0, 1)`
with `gating=false` giving format/rubric credit even when the patch does not apply (non-gating candidate).

## 3. F2P partial-credit candidates (spec §6)
| id | T(frac) | half-passed value (in the partial band) |
|---|---|---|
| rw_v002 | binary | 0 |
| rw_v003 | linear | 0.50 |
| rw_v004 | sqrt | 0.71 |
| rw_v001 | frac^1.5 (baseline) | 0.35 |
| rw_v005 | ladder ≥34 %→0.2, ≥67 %→0.5, 100 %→1 | 0.20 |
Final reward for "half of F2P" = `s_applies + 0.60·T` → 0.10 / 0.40 / 0.52 / 0.31 / 0.22.

## 4. P2P regression penalty candidates (spec §7)
| id | M(k regressions of n) | 1/30 | 2/30 | 5/30 | 30/30 |
|---|---|---|---|---|---|
| rw_v001 | 1−0.75·(k/n)^1.5 (baseline) | 0.995 | 0.987 | 0.949 | 0.25 |
| rw_v006 strong | 1−1.0·(k/n) | 0.967 | 0.933 | 0.833 | 0.00 |
| rw_v007 tolerant | first 2 free, then pow | 1.0 | 1.0 | 0.93 | 0.28 |
| rw_v008 capped | 1−min(0.5, 0.75·k/n) | 0.975 | 0.95 | 0.875 | 0.50 |
| rw_v009 confidence | 1−0.75·wilson_lower(k,n)^1.5 | ≈1.0 | 0.999 | 0.99 | 0.36 |
| rw_v010 none | 1 | 1 | 1 | 1 | 1 |
All multipliers are monotone in k (unit-tested). Because only ≤30 P2P ids are checked and the validation
run prunes ids that fail before the patch, an observed regression is certain evidence.

## 5. Rubric / RUSCA (spec §3, §22)
`rubric_score` = weighted satisfaction of the verifier-checkable criteria (`verifier/rubric.py`). It enters
the reward with `rubric_weight` only while `rusca_stage ∈ {scaffold, transition}`; the stage is derived from
the RUSCA schedule (`verifier/rusca.py`, logistic decay centred at 20 % of steps, gone by ≈24 %). Candidates:
`rw_v012_rubric_heavy` (0.4, never decays), `rw_v013_rusca` (0.2 → 0), baseline (0).

## 6. Length (DDCA / overlong)
`overlong` = DAPO-style soft penalty inside the last `overlong_buffer` tokens (off by default; the agent loop
now passes `valid_response_length` so it is not inert). `answer_match` is pure `resolved`, which is what the
DDCA advantage (`rl/algos.py::grpo_ddca`) uses to form the correct cluster.

## 7. Anti-hacking gates
`invalid_*` states (test/config/verifier edits, path escape, pytest skip/conftest hooks, always-true `__eq__`
canary) → `s_invalid` (0) and are **not** infra-excluded, so the policy is penalised for them.

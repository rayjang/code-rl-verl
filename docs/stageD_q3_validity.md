# Stage D (Qwen3) — is the arm ranking measuring anything?

Written 2026-09-06, after arms Q000–Q008 completed. Every number below is reproducible with
`scripts/arm_significance.py`; the JSON it writes is in `results/arm_significance_q3*.json`.

## 1. What the loop concluded

`scripts/autoresearch.py` keeps the arm with the highest `val_resolved` and rejects the rest, by bare
point comparison with `min_improvement: 0.0`. Its verdict after nine arms:

| arm | val_resolved | loop decision |
|---|---|---|
| exp_Q000_baseline | 0.40375 | keep (baseline) |
| exp_Q007_rubric_heavy | 0.40375 | reject |
| exp_Q004_p2p_strong | 0.40304 | reject |
| exp_Q001_f2p_binary | 0.40250 | reject |
| exp_Q006_rule_only | 0.40250 | reject |
| exp_Q008_rusca | 0.40000 | reject |
| exp_Q005_p2p_tolerant | 0.39875 | reject |
| exp_Q002_f2p_linear | 0.39554 | reject |
| exp_Q003_f2p_ladder | — | reject (run failed) |

Total spread across eight completed arms: **0.008**.

## 2. The ranking metric barely moves during a run

`val_resolved` is the unweighted mean of the two per-track solve rates (SWE 112 rows, unit-test 400
rows, greedy n=1). In the baseline it went **0.3834 → 0.3923 → 0.4037 over the 20-step run: +0.020 in total**.

The arm-to-arm spread (0.008) is 40 % of the entire effect of training (0.020). Twenty steps of RL
move this metric by about two and a half times the gap the loop is trying to resolve between arms —
so it can only separate reward functions whose effect is a large fraction of all the learning that
happened, and nothing subtler.

What *does* move in those 20 steps, SWE track, step 0 → 20:

| metric | step 0 | step 20 | delta |
|---|---|---|---|
| `patch_apply_score` | 0.295 | 0.795 | **+0.500** |
| `gated_out` | 0.705 | 0.205 | **−0.500** |
| `patch_format_score` | 0.607 | 0.866 | +0.259 |
| `f2p_frac` | 0.076 | 0.229 | +0.153 |
| `answer_match` (the solve rate) | 0.027 | 0.0625 | +0.036 |

The policy is learning to emit a patch that *applies* — that metric moves 14× more than the solve
rate. It is not yet learning to make the tests pass: `answer_match` is the last link of that chain
and is still near the floor (0.06 on SWE), where a single instance flipping is worth 0.009 of the
track rate. That granularity is coarser than the differences being ranked.

## 3. Every arm is statistically tied with the baseline on `val_resolved`

Arms are evaluated on the same validation instances, so a paired test is much sharper than comparing
two marginal rates. Unpaired SE of `val_resolved` is 0.0158, i.e. nothing below a ~0.032 gap is
measurable at all; the paired bootstrap tightens that to ~±0.011. Even so:

```
arm                      resolved    delta      95% CI (paired)  McNemar p  discordant
exp_Q001_f2p_binary        0.4025  -0.0012  [-0.0112, +0.0088]      1.000   9 win / 10 lose of 512
exp_Q002_f2p_linear        0.3955  -0.0082  [-0.0227, +0.0050]      0.503   8 win / 12 lose
exp_Q004_p2p_strong        0.4030  -0.0007  [-0.0145, +0.0123]      0.815  10 win /  8 lose
exp_Q005_p2p_tolerant      0.3987  -0.0050  [-0.0175, +0.0075]      0.557  11 win / 15 lose
exp_Q006_rule_only         0.4025  -0.0012  [-0.0100, +0.0075]      1.000   6 win /  7 lose
exp_Q007_rubric_heavy      0.4037  +0.0000  [-0.0100, +0.0100]      1.000   8 win /  8 lose
exp_Q008_rusca             0.4000  -0.0037  [-0.0150, +0.0075]      0.648   8 win / 11 lose
```

No CI excludes zero. Only 13–26 of 512 validation instances change verdict between any arm and the
baseline. **The eight arms did not produce measurably different policies on this metric.**

## 4. On metrics that do move, the baseline is a 3σ outlier — which is the real problem

Re-ranking on `f2p_frac` and `patch_apply_score` appears to overturn the loop's verdict: 2 of 7 and
6 of 7 arms respectively beat the baseline with a 95% CI excluding zero. SWE track, final validation:

| arm | answer_match | f2p_frac | patch_apply | gated_out |
|---|---|---|---|---|
| exp_Q000_baseline | 0.0625 | **0.2294** | **0.7946** | **0.2054** |
| exp_Q001_f2p_binary | 0.0625 | 0.2545 | 0.8750 | 0.1250 |
| exp_Q002_f2p_linear | 0.0536 | 0.2494 | 0.8750 | 0.1250 |
| exp_Q004_p2p_strong | 0.0536 | 0.2417 | 0.8661 | 0.1339 |
| exp_Q005_p2p_tolerant | 0.0625 | 0.2409 | 0.8214 | 0.1786 |
| exp_Q006_rule_only | 0.0625 | 0.2588 | 0.8750 | 0.1250 |
| exp_Q007_rubric_heavy | 0.0625 | 0.2505 | 0.8929 | 0.1071 |
| exp_Q008_rusca | 0.0625 | 0.2526 | 0.8482 | 0.1518 |

The baseline sits **below every one of the seven arms** on `f2p_frac`, `patch_apply_score` and
`gated_out` — z = −3.1, −3.0 and +3.0 against the arm distribution.

Seven reward configurations that disagree with each other (binary vs linear F2P, strong vs tolerant
P2P, no rubric vs heavy rubric, RUSCA scheduling) do not plausibly all confer the same benefit. The
parsimonious reading is that **the baseline run is the outlier**, and the apparent "arm effects" are
that one run being low. Two concrete reasons this is not paranoia:

1. **The baseline did not run the arms' protocol.** Diffing the resolved overrides, the baseline
   differs from every arm in exactly two lines:

   ```
   trainer.test_freq=10          vs  trainer.test_freq=20
   trainer.val_before_train=True vs  trainer.val_before_train=False
   ```

   The baseline performed three full 512-instance greedy validation passes (steps 0, 10, 20); each
   arm performed one (step 20). This is a leftover from the baseline being hand-debugged over five
   failed attempts while the arms were generated from `experiments/base_stageD_q3/overrides.txt`.
   The control was not run under the treatment's protocol.

2. **There is one run per arm, so run-to-run variance is unmeasured.** Every arm-vs-baseline
   comparison shares the same single baseline run, so they are not independent: one low baseline
   shifts all seven deltas together, exactly the pattern observed. The paired bootstrap in §3 prices
   in *instance* sampling noise but not *training-run* noise, and only the latter can explain a
   uniform shift.

## 5. What is being run about it

Three control replicates, identical to the arm protocol (`val_before_train=False`, `test_freq=20`,
baseline reward `rw_v001_baseline.yaml`), submitted 2026-09-06:

| experiment | seed | measures |
|---|---|---|
| `exp_Q000r1_ctrl` | 0 | run-to-run nondeterminism at the arms' seed |
| `exp_Q000r2_ctrl` | 0 | " |
| `exp_Q000r3_seed1` | 1 | additional seed sensitivity |

All arms ran at `data.seed=0`, so r1/r2 isolate exactly the noise that separates two arms: vLLM
sampling, FSDP reduction order, verifier container timing. Their spread on `f2p_frac` and
`patch_apply_score` is the discrimination threshold every Stage-D claim has to clear.

Predictions, stated before the runs land:

* If r1/r2/r3 land in the arms' band (`f2p_frac` ≈ 0.24–0.26, apply ≈ 0.82–0.89), then the
  §4 "arm effects" are run noise, the baseline was simply a low draw, and **no reward configuration
  tested so far is distinguishable from any other** — the honest Stage-D result.
* If r1/r2/r3 reproduce the baseline (`f2p_frac` ≈ 0.229, apply ≈ 0.795), the gap is real and the
  ranking on `f2p_frac`/`patch_apply_score` in §4 stands, with `rubric_heavy` and `rule_only` ahead.

## 6. Consequences for the plan regardless of outcome

1. **Change the primary metric.** `val_resolved` has no dynamic range at 20 steps. Rank on
   `f2p_frac` (SWE), which moves 3× and is continuous, so it carries far more statistical power per
   run. Keep `answer_match` as a reported outcome, not as the selector.
2. **Set `min_improvement` from the measured replicate spread** instead of `0.0`. A bare
   `mine > best` comparison on n=1 runs will keep whichever arm drew the luckiest seed.
3. **Re-run the baseline under the arm protocol** — done, above; `exp_Q000_baseline` should not be
   used as the control for the Q arms.
4. Only measurements independent of the reward config may be compared across arms: `answer_match`,
   `f2p_frac`, `patch_apply_score`, `p2p_frac`, `gated_out`. `final_reward` is **not** comparable —
   each arm computes it with a different reward function, so a "higher reward" arm may simply have a
   more generous reward.
5. 20 steps may be too short for reward shaping to separate at all, given the policy is still on the
   "make the patch apply" part of the curve. If the replicates show the arms are indistinguishable,
   the next lever is a longer run on fewer arms, not more arms at 20 steps.

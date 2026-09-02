import itertools, math
import pytest
from verifier.reward import RewardConfig, compute_reward, baseline_score_of
from verifier.schemas import (ApplyResult, ApplyStatus, ErrKind, ExecutionResult, ExtractStatus, ExtractionResult, TestOutcome)
from verifier import f2p, p2p


def ext(status=ExtractStatus.DIFF_PARSE_SUCCESS):
    return ExtractionResult(status, "diff --git a/x b/x\n" if status != ExtractStatus.NO_PATCH else "", "fenced_diff", "markdown", 1, "", ("x",))


def ran(nf, tf, np_, tp):
    return ExecutionResult(True, ErrKind.OK, "", TestOutcome(tuple(f"f{i}" for i in range(nf)), tuple(f"f{i}" for i in range(nf, tf))),
                           TestOutcome(tuple(f"p{i}" for i in range(np_)), tuple(f"p{i}" for i in range(np_, tp))),
                           apply=ApplyResult(ApplyStatus.PATCH_APPLY_SUCCESS))


def test_ladder_default_reproduces_baseline_table():
    cfg = RewardConfig()
    for nf, tf, np_, tp in itertools.product([0, 2, 5, 8, 9, 10], [10], [0, 15, 29, 30], [30]):
        er = ran(nf, tf, np_, tp)
        rc = compute_reward(ext(), er, cfg)
        assert math.isclose(rc.final_reward, baseline_score_of("ran", nf / tf, np_ / tp), abs_tol=1e-9), (nf, np_)
    assert compute_reward(ext(ExtractStatus.NO_PATCH), ExecutionResult(False, ErrKind.FORMAT_NO_PATCH), cfg).final_reward == 0.0
    er = ExecutionResult(False, ErrKind.FORMAT_APPLY_FAIL, "patch_apply_fail", apply=ApplyResult(ApplyStatus.PATCH_APPLY_FAIL))
    assert compute_reward(ext(), er, cfg).final_reward == 0.05


def test_documented_values():
    cfg = RewardConfig()
    assert math.isclose(compute_reward(ext(), ran(5, 10, 30, 30), cfg).final_reward, 0.312, abs_tol=2e-3)
    assert math.isclose(compute_reward(ext(), ran(10, 10, 29, 30), cfg).final_reward, 0.846, abs_tol=2e-3)
    assert math.isclose(compute_reward(ext(), ran(10, 10, 0, 30), cfg).final_reward, 0.212, abs_tol=2e-3)
    assert compute_reward(ext(), ran(10, 10, 30, 30), cfg).final_reward == 1.0


def test_infra_excluded_and_invalid():
    cfg = RewardConfig()
    rc = compute_reward(ext(), ExecutionResult(False, ErrKind.INFRA_TIMEOUT, "timeout"), cfg)
    assert rc.infra_excluded == 1.0 and rc.final_reward == 0.0 and rc.err_kind == "infra_timeout"
    rc = compute_reward(ext(), ExecutionResult(False, ErrKind.INVALID_EDITS_TESTS, "x"), cfg)
    assert rc.infra_excluded == 0.0 and rc.final_reward == 0.0 and rc.state == "invalid"


def test_parse_fail_distinct_from_no_patch_and_apply_fail():
    cfg = RewardConfig(s_parse_fail=0.02)
    rc = compute_reward(ext(ExtractStatus.DIFF_PARSE_FAIL), ExecutionResult(False, ErrKind.FORMAT_PARSE_FAIL), cfg)
    assert rc.final_reward == 0.02 and rc.patch_extraction_score == 1.0 and rc.patch_format_score == 0.0


def test_f2p_modes_monotone_and_half():
    for mode in ("binary", "linear", "sqrt", "pow", "ladder"):
        vals = [f2p.transform(x / 10, mode) for x in range(11)]
        assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])), mode
        assert vals[-1] == 1.0
    assert f2p.transform(0.5, "binary") == 0.0 and f2p.transform(0.5, "linear") == 0.5
    assert math.isclose(f2p.transform(0.5, "sqrt"), 0.7071, abs_tol=1e-3) and math.isclose(f2p.transform(0.5, "pow"), 0.3536, abs_tol=1e-3)
    assert f2p.transform(0.5, "ladder") == 0.2 and f2p.transform(0.7, "ladder") == 0.5


def test_p2p_modes():
    assert p2p.multiplier(30, 30) == 1.0
    assert math.isclose(p2p.multiplier(29, 30, "pow"), 0.9954, abs_tol=1e-3)
    assert math.isclose(p2p.multiplier(28, 30, "tolerance", tolerance=2), 1.0)
    assert p2p.multiplier(0, 30, "capped", cap=0.5) == 0.5
    assert p2p.multiplier(29, 30, "confidence") > p2p.multiplier(29, 30, "linear")   # certain part only
    assert p2p.multiplier(10, 30, "catastrophic", catastrophic_frac=0.5) == 0.25
    assert p2p.multiplier(29, 30, "catastrophic", tolerance=1) == 1.0
    for mode in ("linear", "pow", "capped", "tolerance", "confidence", "catastrophic"):
        vals = [p2p.multiplier(k, 30, mode) for k in range(31)]
        assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])), mode


def test_additive_nongating_gives_format_credit_on_apply_fail():
    cfg = RewardConfig(mode="additive", gating=False, rubric_weight=0.2, rusca_stage="scaffold")
    er = ExecutionResult(False, ErrKind.FORMAT_APPLY_FAIL, "patch_apply_fail", apply=ApplyResult(ApplyStatus.PATCH_APPLY_FAIL))
    rc = compute_reward(ext(), er, cfg, rubric_score=0.5)
    assert math.isclose(rc.final_reward, 0.05 + 0.2 * 0.5, abs_tol=1e-9)
    cfg2 = RewardConfig(mode="additive", gating=True, rubric_weight=0.2, rusca_stage="scaffold")
    assert math.isclose(compute_reward(ext(), er, cfg2, rubric_score=0.5).final_reward, 0.05, abs_tol=1e-9)


def test_rubric_ignored_in_rule_stage():
    cfg = RewardConfig(rubric_weight=0.3, rusca_stage="rule")
    a = compute_reward(ext(), ran(5, 10, 30, 30), cfg, rubric_score=1.0).final_reward
    b = compute_reward(ext(), ran(5, 10, 30, 30), RewardConfig()).final_reward
    assert a == b


def test_overlong_penalty():
    cfg = RewardConfig(overlong_buffer=100, overlong_penalty=1.0, max_response_length=1000)
    rc = compute_reward(ext(), ran(10, 10, 30, 30), cfg, response_len=950)
    assert math.isclose(rc.final_reward, 0.5) and rc.score_before_overlong == 1.0 and rc.answer_match == 1.0


def test_yaml_roundtrip(tmp_path):
    import yaml
    cfg = RewardConfig(version="rw_test", f2p_mode="ladder")
    p = tmp_path / "r.yaml"; p.write_text(yaml.safe_dump(cfg.dump()))
    assert RewardConfig.load(str(p)) == cfg
